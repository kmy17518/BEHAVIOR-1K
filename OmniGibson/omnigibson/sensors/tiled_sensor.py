import math
import os
import omnigibson as og
import omnigibson.lazy as lazy
import torch as th
import warp as wp
from omnigibson.sensors.vision_sensor import VisionSensor
from omnigibson.utils.ui_utils import create_module_logger
from typing import Any, Dict, List, Optional, Tuple

# Create module logger
log = create_module_logger(module_name=__name__)


# Vendored from isaacsim.sensors.camera.camera_view.reshape_tiled_image, with the unused offset /
# num_output_channels arguments dropped and a uint32 overload added -- upstream only registers uint8
# and float32, but non-colored segmentation annotators hand back raw uint32 IDs.
@wp.kernel(enable_backward=False)
def reshape_tiled_image(
    tiled_image_buffer: Any,
    batched_image: Any,
    image_height: int,
    image_width: int,
    num_channels: int,
    num_tiles_x: int,
):
    """Reshapes a tiled image into a batch of images.
    This function reshapes the input tiled image buffer into a batch of images. The input image buffer
    is assumed to be tiled in the x and y directions. The output image is a batch of images with the
    specified height, width, and number of channels.
    Args:
        tiled_image_buffer: The input image buffer. Shape is (height * width * num_channels * num_cameras,).
        batched_image: The output image. Shape is (num_cameras, height, width, num_channels).
        image_width: The width of the image.
        image_height: The height of the image.
        num_channels: The number of channels in the image.
        num_tiles_x: The number of tiles in x-direction.
    """
    # get the thread id
    camera_id, height_id, width_id = wp.tid()

    # resolve the tile indices
    tile_x_id = camera_id % num_tiles_x
    tile_y_id = camera_id // num_tiles_x
    # compute the start index of the pixel in the tiled image buffer
    pixel_start = (
        num_channels * num_tiles_x * image_width * (image_height * tile_y_id + height_id)
        + num_channels * tile_x_id * image_width
        + num_channels * width_id
    )

    # copy the pixel values into the batched image
    for i in range(num_channels):
        batched_image[camera_id, height_id, width_id, i] = tiled_image_buffer[pixel_start + i]


# uint32 is used for non-colored segmentation annotators (raw IDs, no dtype conversion is performed)
wp.overload(
    reshape_tiled_image,
    {"tiled_image_buffer": wp.array(dtype=wp.uint32), "batched_image": wp.array(dtype=wp.uint32, ndim=4)},
)
# uint8 is used for 4 channel annotators
wp.overload(
    reshape_tiled_image,
    {"tiled_image_buffer": wp.array(dtype=wp.uint8), "batched_image": wp.array(dtype=wp.uint8, ndim=4)},
)
# float32 is used for single channel annotators
wp.overload(
    reshape_tiled_image,
    {"tiled_image_buffer": wp.array(dtype=wp.float32), "batched_image": wp.array(dtype=wp.float32, ndim=4)},
)


class TiledVisionSensor:
    """
    Tiled vision sensor that leverages Omniverse Replicator to render tiled images from multiple cameras
    Args:
        envs (Iterable): Robot containers to create tiled sensors for -- any objects exposing a .robots list
            (e.g. the per-env Scene instances of a multi-env Environment), one per environment, in env order.
        tile_all_sensors (bool): Whether to tile all cameras in the envs or only those with matching names. Defaults to False.
            NOTE: If True, all cameras must have the same modalities and resolution.
    """

    # Registry of all created tiled sensors, used for cleanup on og.clear()
    TILED_SENSORS = []

    def __init__(self, envs: List[Any], tile_all_sensors: bool = False, external_sensors: Optional[List[Dict[str, Any]]] = None) -> None:
        camera_prim_paths = dict()
        self.modalities = dict()
        self._camera_resolution = dict()
        self._camera_prims = dict()
        self._annotators = dict()
        # iSpatialGym vec-eval overlay (2026-10-29): ISG_VEC_TILED_GROUP=clone groups the tiles **per env** instead of per camera name
        # - one render product per env holding that env's equal-resolution, equal-modality robot cameras (the R1Pro's two 480^2
        # wrists; a group of one camera stays on its per-camera product, so the 720^2 head is untouched). Reason: in a tiled
        # product the screen-space / temporal post passes (DLSS, AO, denoisers) are computed with the product's single camera,
        # and a tile's error grows with its camera's offset from that camera (0 at 0 m, -8..-12 % at 57 m, -25 % at 114 m,
        # campaign vec_speed): cameras of one robot are within ~1 m of each other, cameras of different clones 50-150 m apart.
        # "name" (the default) is the upstream behaviour. The lookup API (tile_of / covers / slice) serves both modes.
        self.group_mode = os.environ.get("ISG_VEC_TILED_GROUP", "name")
        self._tile_of = dict()  # (robot_name, sensor_name, env_idx) -> (group key, tile index)
        self._tiled_sensor_names = set()  # sensor names served by a tiled product (every env)
        self._tiled_external_names = set()  # external sensor names served by a tiled product (vec-eval overlay, 2026-10-29)
        # modes: "name" | "clone" for the robot cameras, "+external" adds one product per external camera name across the envs
        # (the video's over-the-shoulder view: no policy consumes it, so the non-primary tiles' post-processing error is cosmetic)
        modes = {m.strip() for m in self.group_mode.split("+") if m.strip()}
        unknown = modes - {"name", "clone", "external"}
        assert not unknown, f"ISG_VEC_TILED_GROUP: unknown mode(s) {sorted(unknown)} in {self.group_mode!r}"
        robot_mode = "clone" if "clone" in modes else ("name" if "name" in modes else None)
        # ISG_VEC_TILED_SENSORS (2026-10-29): comma list of substrings; in "name" mode only robot sensors whose name contains one of
        # them are tiled (e.g. "zed_link" = the head only -> a single tiled resolution in the session; the Replicator docstring's
        # single-resolution note is tested with it). Empty = every vision sensor, as upstream.
        only = [t for t in os.environ.get("ISG_VEC_TILED_SENSORS", "").split(",") if t.strip()]

        stage = lazy.omni.usd.get_context().get_stage()
        if robot_mode == "clone":
            # pass 1: per env, per robot, the vision sensors by (resolution, modalities); pass 2: groups with >= 2 cameras become products
            for env_idx, env in enumerate(envs):
                for robot in env.robots:
                    buckets = dict()
                    for sensor_name, sensor in robot.sensors.items():
                        if isinstance(sensor, VisionSensor):
                            buckets.setdefault(((sensor.image_width, sensor.image_height), tuple(sorted(sensor.modalities))), []).append((sensor_name, sensor))
                    for (res, _mods), members in buckets.items():
                        if len(members) < 2:
                            continue
                        key = f"env{env_idx}:{robot.name}:{res[0]}x{res[1]}"
                        camera_prim_paths[key] = []
                        self.modalities[key] = members[0][1].modalities
                        self._camera_resolution[key] = res
                        self._camera_prims[key] = []
                        for tile_idx, (sensor_name, sensor) in enumerate(sorted(members, key=lambda m: m[0])):
                            camera_prim_paths[key].append(sensor.prim_path)
                            self._camera_prims[key].append(lazy.pxr.UsdGeom.Camera(stage.GetPrimAtPath(sensor.prim_path)))
                            self._tile_of[(robot.name, sensor_name, env_idx)] = (key, tile_idx)
                            self._tiled_sensor_names.add(sensor_name)
            log.info(f"TiledVisionSensor (clone grouping): {len(camera_prim_paths)} products, sensors {sorted(self._tiled_sensor_names)}")
        elif robot_mode == "name":
            for env_idx, env in enumerate(envs):
                for robot in env.robots:
                    for sensor_name, sensor in robot.sensors.items():
                        if only and not any(t in sensor_name for t in only):
                            continue
                        key = "sensor" if tile_all_sensors else sensor_name
                        if isinstance(sensor, VisionSensor):
                            if key not in camera_prim_paths:
                                camera_prim_paths[key] = []
                                self.modalities[key] = sensor.modalities
                                self._camera_resolution[key] = (sensor.image_width, sensor.image_height)
                                self._camera_prims[key] = []
                            assert (
                                sensor.modalities == self.modalities[key]
                            ), f"All sensors named {key} must have the same modalities!"
                            assert self._camera_resolution[key] == (
                                sensor.image_width,
                                sensor.image_height,
                            ), f"All sensors named {key} must have the same resolution!"
                            self._tile_of[(robot.name, sensor_name, env_idx)] = (key, len(camera_prim_paths[key]))
                            self._tiled_sensor_names.add(sensor_name)
                            camera_prim_paths[key].append(sensor.prim_path)
                            camera_prim = stage.GetPrimAtPath(sensor.prim_path)
                            self._camera_prims[key].append(lazy.pxr.UsdGeom.Camera(camera_prim))
        if "external" in modes and external_sensors:
            # one tiled product per external camera name across the envs (equal resolution / modalities required, as upstream)
            for env_idx, sensors_of_env in enumerate(external_sensors):
                for sensor_name, sensor in sensors_of_env.items():
                    if not isinstance(sensor, VisionSensor):
                        continue
                    key = f"external:{sensor_name}"
                    if key not in camera_prim_paths:
                        camera_prim_paths[key] = []
                        self.modalities[key] = sensor.modalities
                        self._camera_resolution[key] = (sensor.image_width, sensor.image_height)
                        self._camera_prims[key] = []
                    assert sensor.modalities == self.modalities[key], f"All external sensors named {sensor_name} must have the same modalities!"
                    assert self._camera_resolution[key] == (sensor.image_width, sensor.image_height), f"All external sensors named {sensor_name} must have the same resolution!"
                    self._tile_of[("external", sensor_name, env_idx)] = (key, len(camera_prim_paths[key]))
                    self._tiled_external_names.add(sensor_name)
                    camera_prim_paths[key].append(sensor.prim_path)
                    self._camera_prims[key].append(lazy.pxr.UsdGeom.Camera(stage.GetPrimAtPath(sensor.prim_path)))
            log.info(f"TiledVisionSensor (external grouping): {sorted(self._tiled_external_names)} tiled across {len(external_sensors)} envs")
        self._render_product_paths = {
            key: lazy.omni.replicator.core.create.render_product_tiled(
                cameras=camera_prim_paths[key], tile_resolution=self._camera_resolution[key]
            )
            for key in camera_prim_paths
        }
        self._create_annotators()
        # Attach the annotator to the render product
        for sensor_name in self._annotators:
            for modality in self._annotators[sensor_name]:
                self._annotators[sensor_name][modality].attach([self._render_product_paths[sensor_name]])

        # Create internal buffers
        self._create_buffers()

        # Register for cleanup on og.clear()
        TiledVisionSensor.TILED_SENSORS.append(self)

    def remove(self) -> None:
        """Detach all annotators and destroy the tiled render products. Idempotent."""
        if self not in TiledVisionSensor.TILED_SENSORS:
            return
        TiledVisionSensor.TILED_SENSORS.remove(self)
        with og.sim.editing_usd():
            for sensor_name in self._annotators:
                for annotator in self._annotators[sensor_name].values():
                    # Passing an explicit list is bugged -- see VisionSensor._remove_modality_from_backend
                    try:
                        annotator.detach(self._render_product_paths[sensor_name])
                    except TypeError:
                        log.warning(f"Failed to cleanly detach tiled annotator for {sensor_name}; skipping")
        # Render so the syntheticdata graph settles before the render product is destroyed -- destroying
        # with pending graph edits leaves invalid nodes that break subsequent annotator detach calls
        og.sim.render()
        with og.sim.editing_usd():
            for render_product in self._render_product_paths.values():
                render_product.destroy()
        og.sim.render()
        self._annotators = dict()
        self._render_product_paths = dict()

    @classmethod
    def clear(cls) -> None:
        """Remove all tiled vision sensors. Called on og.clear() before scenes (and their cameras) are removed."""
        for sensor in tuple(cls.TILED_SENSORS):
            sensor.remove()

    def _create_annotators(self) -> None:
        self._annotators = dict()
        for sensor_name in self.modalities:
            self._annotators[sensor_name] = dict()
            for modality in self.modalities[sensor_name]:
                annotator_type = VisionSensor.RAW_SENSOR_TYPES[modality]
                annotator = lazy.omni.replicator.core.AnnotatorRegistry.get_annotator(
                    annotator_type, device="cuda:0", do_array_copy=False
                )
                self._annotators[sensor_name][modality] = annotator

    def _create_buffers(self, device: str = "cuda:0") -> None:
        self._output_buffer = dict()
        self._output_warp_buffer = dict()
        self._reshape_dims = dict()
        self._num_tiles_x = dict()
        for sensor_name in self._camera_prims:
            self._output_buffer[sensor_name] = dict()
            self._output_warp_buffer[sensor_name] = dict()
            self._reshape_dims[sensor_name] = (
                self._camera_count(sensor_name=sensor_name),
                self._camera_resolution[sensor_name][1],
                self._camera_resolution[sensor_name][0],
            )
            self._num_tiles_x[sensor_name] = self._tiled_grid_shape(sensor_name=sensor_name)[0]
            for modality in self.modalities[sensor_name]:
                if modality == "rgb":
                    self._output_buffer[sensor_name][modality] = th.zeros(
                        (
                            self._camera_count(sensor_name=sensor_name),
                            self._camera_resolution[sensor_name][1],
                            self._camera_resolution[sensor_name][0],
                            4,
                        ),
                        device=device,
                        dtype=th.uint8,
                    ).contiguous()
                elif modality == "depth" or modality == "depth_linear":
                    self._output_buffer[sensor_name][modality] = th.zeros(
                        (
                            self._camera_count(sensor_name=sensor_name),
                            self._camera_resolution[sensor_name][1],
                            self._camera_resolution[sensor_name][0],
                            1,
                        ),
                        device=device,
                        dtype=th.float32,
                    ).contiguous()
                elif modality == "seg_semantic" or modality == "seg_instance" or modality == "seg_instance_id":
                    # unlike VisionSensor.get_obs(), the tiled path returns RAW Replicator segmentation
                    # IDs -- they are arbitrary, change between sessions, and are NOT remapped to OmniGibson's
                    # stable semantic/instance IDs (see VisionSensor._remap_semantic_segmentation). No id->name
                    # mapping info dict is available either. Do NOT train on these values until a batched remap
                    # is implemented; use the single-env per-sensor path for correct segmentation labels.
                    self._output_buffer[sensor_name][modality] = th.zeros(
                        (
                            self._camera_count(sensor_name=sensor_name),
                            self._camera_resolution[sensor_name][1],
                            self._camera_resolution[sensor_name][0],
                            1,
                        ),
                        device=device,
                        dtype=th.uint32,
                    ).contiguous()
                elif modality == "flow":
                    self._output_buffer[sensor_name][modality] = th.zeros(
                        (
                            self._camera_count(sensor_name=sensor_name),
                            self._camera_resolution[sensor_name][1],
                            self._camera_resolution[sensor_name][0],
                            2,
                        ),
                        device=device,
                        dtype=th.float32,
                    ).contiguous()
                elif modality == "normal":
                    self._output_buffer[sensor_name][modality] = th.zeros(
                        (
                            self._camera_count(sensor_name=sensor_name),
                            self._camera_resolution[sensor_name][1],
                            self._camera_resolution[sensor_name][0],
                            3,
                        ),
                        device=device,
                        dtype=th.uint8,
                    ).contiguous()
                else:
                    raise ValueError(f"Unsupported modality {modality} for tiled vision sensor!")
                # The output tensors are persistent. Cache their zero-copy Warp aliases instead of
                # recreating Python/Warp wrappers on every rendered frame.
                self._output_warp_buffer[sensor_name][modality] = lazy.warp.from_torch(
                    self._output_buffer[sensor_name][modality]
                )

    def _camera_count(self, sensor_name: str) -> int:
        return len(self._camera_prims[sensor_name])

    # -- lookup API (vec-eval overlay): where a robot sensor's tile lives, whatever the grouping mode --------------------
    def covers(self, robot_name: str, sensor_name: str, env_idx: int) -> bool:
        """Is this env's sensor served by a tiled product (else it keeps its per-camera product)?"""
        return (robot_name, sensor_name, env_idx) in self._tile_of

    def covers_sensor(self, sensor_name: str) -> bool:
        return sensor_name in self._tiled_sensor_names

    def covers_external(self, sensor_name: str) -> bool:
        """Is this external (per-env, non-robot) sensor served by a tiled product? (slice with robot_name="external")"""
        return sensor_name in self._tiled_external_names

    @property
    def num_products(self) -> int:
        return len(self._render_product_paths)

    def tile_of(self, robot_name: str, sensor_name: str, env_idx: int) -> Tuple[str, int]:
        return self._tile_of[(robot_name, sensor_name, env_idx)]

    def slice(self, tiled_obs, robot_name: str, sensor_name: str, env_idx: int):
        """{modality: tensor} of one env's sensor out of the batched buffers of :meth:`get_obs` (single-channel modalities (H, W))."""
        key, tile = self._tile_of[(robot_name, sensor_name, env_idx)]
        out = dict()
        for modality in self.modalities[key]:
            data = tiled_obs[key][modality][tile]
            if data.shape[-1] == 1:
                data = data.squeeze(-1)
            out[modality] = data
        return out

    def _tiled_grid_shape(self, sensor_name: str) -> Tuple[int, int]:
        cols = math.ceil(math.sqrt(self._camera_count(sensor_name=sensor_name)))
        rows = math.ceil(self._camera_count(sensor_name=sensor_name) / cols)
        return (cols, rows)

    def _tiled_img_shape(self, sensor_name: str) -> Tuple[int, int]:
        cols, rows = self._tiled_grid_shape(sensor_name=sensor_name)
        width, height = self._camera_resolution[sensor_name]
        return (width * cols, height * rows)

    def get_obs(self):
        for sensor_name in self._annotators:
            for modality in self.modalities[sensor_name]:
                tiled_data_buffer = self._annotators[sensor_name][modality].get_data().to("cuda:0")
                # For flow, we only require the first two channels of the tiled buffer
                # Note: Not doing this breaks the alignment of the data (check: https://github.com/isaac-sim/IsaacLab/issues/2003)
                if modality == "flow":
                    tiled_data_buffer = tiled_data_buffer[:, :, :2].contiguous()
                lazy.warp.launch(
                    kernel=reshape_tiled_image,
                    dim=self._reshape_dims[sensor_name],
                    inputs=[
                        tiled_data_buffer.flatten(),
                        self._output_warp_buffer[sensor_name][modality],
                        *list(self._output_buffer[sensor_name][modality].shape[1:]),  # height, width, num_channels
                        self._num_tiles_x[sensor_name],
                    ],
                    device="cuda:0",
                )

        return self._output_buffer
