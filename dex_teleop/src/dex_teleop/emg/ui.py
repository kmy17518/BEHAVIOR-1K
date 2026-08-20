"""Native Omniverse UI for a bounded live EMG preview."""

from __future__ import annotations

import time

import numpy as np

from dex_teleop.emg.session import EmgSession


PLOT_UPDATE_PERIOD_S = 0.1
PLOT_WINDOW_SECONDS = 1.0
PLOT_MINIMUM_RANGE_UV = 1.0
EMG_DOCK_RATIO = 0.5
DECODER_DOCK_RATIO = 0.5
IMPEDANCE_GOOD_MAX_OHM = 500_000.0
IMPEDANCE_FAIR_MAX_OHM = 999_000.0
IMPEDANCE_GOOD_COLOR = 0xFF59A14F
IMPEDANCE_FAIR_COLOR = 0xFFF28E2B
IMPEDANCE_POOR_COLOR = 0xFFE15759
IMPEDANCE_UNKNOWN_COLOR = 0xFFAAAAAA
CHANNEL_COLORS = (
    0xFF4E79A7,
    0xFFF28E2B,
    0xFFE15759,
    0xFF76B7B2,
    0xFF59A14F,
    0xFFEDC948,
    0xFFB07AA1,
    0xFFFF9DA7,
)


def _dock_position(ui, position: str):
    try:
        return getattr(ui.DockPosition, position.upper())
    except AttributeError as error:
        raise ValueError(f"Unsupported Kit dock position {position!r}") from error


def impedance_display(impedance_ohm: float) -> tuple[str, int]:
    """Format one SDK impedance using the example GUI's contact thresholds."""

    if not np.isfinite(impedance_ohm) or impedance_ohm <= 0.0:
        return "— kOhm", IMPEDANCE_UNKNOWN_COLOR
    if impedance_ohm <= IMPEDANCE_GOOD_MAX_OHM:
        color = IMPEDANCE_GOOD_COLOR
    elif impedance_ohm <= IMPEDANCE_FAIR_MAX_OHM:
        color = IMPEDANCE_FAIR_COLOR
    else:
        color = IMPEDANCE_POOR_COLOR
    return f"{impedance_ohm / 1000.0:.0f} kOhm", color


class EmgMonitor:
    """Update Kit widgets only when called from the simulator's main thread."""

    def __init__(
        self,
        session: EmgSession,
        *,
        dock_parent_name: str,
        dock_position: str = "right",
        dock_ratio: float = EMG_DOCK_RATIO,
        visualize_decoder: bool = False,
        decoder_hand: str = "right",
        decoder_dock_parent_name: str | None = None,
        decoder_dock_position: str = "bottom",
        decoder_dock_ratio: float = DECODER_DOCK_RATIO,
    ) -> None:
        import omnigibson.lazy as lazy
        from omnigibson.utils.ui_utils import dock_window

        self._session = session
        self._ui = lazy.omni.ui
        self._last_update = 0.0
        self._plots = []
        self._impedance_labels = []
        self._window = self._ui.Window("OYMotion EMG", width=900, height=500)
        with self._window.frame:
            with self._ui.VStack(spacing=3):
                self._status_label = self._ui.Label("EMG: waiting for samples", height=24)
                for channel in range(session.channel_count):
                    with self._ui.HStack(height=48, spacing=4):
                        self._ui.Label(f"Ch {channel + 1}", width=44)
                        plot = self._ui.Plot(
                            self._ui.Type.LINE,
                            -PLOT_MINIMUM_RANGE_UV,
                            PLOT_MINIMUM_RANGE_UV,
                            0.0,
                            height=44,
                            style={
                                "color": CHANNEL_COLORS[channel % len(CHANNEL_COLORS)],
                                "background_color": 0xFF161616,
                            },
                        )
                        self._plots.append(plot)
                        impedance_label = self._ui.Label(
                            "— kOhm",
                            width=84,
                            style={"color": IMPEDANCE_UNKNOWN_COLOR},
                        )
                        self._impedance_labels.append(impedance_label)
        dock_window(
            space=self._ui.Workspace.get_window(dock_parent_name),
            name=self._window.title,
            location=_dock_position(self._ui, dock_position),
            ratio=dock_ratio,
        )
        self._decoder_monitor = (
            DecodedHandMonitor(
                session,
                dock_parent_name=decoder_dock_parent_name or self._window.title,
                dock_position=decoder_dock_position,
                dock_ratio=decoder_dock_ratio,
                hand=decoder_hand,
            )
            if visualize_decoder
            else None
        )

    def update(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_update < PLOT_UPDATE_PERIOD_S:
            return
        self._last_update = now
        self._session.check_health()
        sample_limit = max(1, int(round(self._session.sample_rate_hz * PLOT_WINDOW_SECONDS)))
        preview = self._session.preview(max_samples=sample_limit)
        if len(preview.sample_index) == 0:
            self._status_label.text = f"EMG: {preview.status}"
            return

        lost_count = int(np.count_nonzero(preview.is_lost))
        device_name = self._session.metadata.get("device_name", "OYMotion")
        self._status_label.text = (
            f"{device_name} | {self._session.channel_count} ch @ {self._session.sample_rate_hz:g} Hz | "
            f"sample {int(preview.sample_index[-1])} | lost in view: {lost_count}"
        )
        channel_count = min(preview.signal_uv.shape[1], len(self._plots))
        for channel in range(channel_count):
            impedance_text, impedance_color = impedance_display(float(preview.impedance_ohm[channel]))
            impedance_label = self._impedance_labels[channel]
            impedance_label.text = impedance_text
            impedance_label.style = {"color": impedance_color}
            values = preview.signal_uv[:, channel]
            finite_values = values[np.isfinite(values)]
            if len(finite_values):
                scale = max(PLOT_MINIMUM_RANGE_UV, float(np.percentile(np.abs(finite_values), 99.0)) * 1.1)
            else:
                scale = PLOT_MINIMUM_RANGE_UV
            plot = self._plots[channel]
            plot.scale_min = -scale
            plot.scale_max = scale
            plot.set_data(*values.tolist())
        if self._decoder_monitor is not None:
            self._decoder_monitor.update(force=force)

    def close(self) -> None:
        if self._window is None:
            return
        if self._decoder_monitor is not None:
            self._decoder_monitor.close()
            self._decoder_monitor = None
        self._window.visible = False
        self._window.destroy()
        self._window = None


class DecodedHandMonitor:
    """Render the sidecar's decoded UmeTrack mesh in native Kit SceneUI."""

    def __init__(
        self,
        session: EmgSession,
        *,
        dock_parent_name: str,
        hand: str,
        dock_position: str = "bottom",
        dock_ratio: float = DECODER_DOCK_RATIO,
    ) -> None:
        import omnigibson.lazy as lazy
        from omnigibson.utils.ui_utils import dock_window

        self._session = session
        self._ui = lazy.omni.ui
        self._scene = lazy.omni.ui.scene
        self._hand = hand
        self._last_update = 0.0
        self._displayed_sequence = -1
        self._topology_signature = None
        self._center = None
        self._radius = None
        self._window = self._ui.Window("EMG2Pose Hand", width=900, height=500)

        projection = [1.7, 0, 0, 0, 0, 3, 0, 0, 0, 0, -1, -1, 0, 0, -2, 0]
        rotation = self._scene.Matrix44.get_rotation_matrix(18, -105, 0, True)
        translation = self._scene.Matrix44.get_translation_matrix(0, 0, -3.2)
        camera = self._scene.CameraModel(projection, translation * rotation)
        placeholder_vertices = [[-0.3, -0.3, 0.0], [0.3, -0.3, 0.0], [0.0, 0.3, 0.0]]
        placeholder_colors = [[0.95, 0.55, 0.66, 1.0]] * 3
        with self._window.frame:
            with self._ui.VStack(spacing=3):
                self._status_label = self._ui.Label("EMG2Pose: loading decoder", height=24)
                with self._ui.ZStack():
                    self._ui.Rectangle(style={"background_color": 0xFF101014})
                    self._scene_view = self._scene.SceneView(
                        camera,
                        aspect_ratio_policy=self._scene.AspectRatioPolicy.PRESERVE_ASPECT_FIT,
                    )
                    with self._scene_view.scene:
                        self._surface = self._scene.PolygonMesh(
                            placeholder_vertices,
                            placeholder_colors,
                            [3],
                            [0, 1, 2],
                        )
                        self._wireframe = self._scene.PolygonMesh(
                            placeholder_vertices,
                            [[0.35, 0.10, 0.16, 0.8]] * 3,
                            [3],
                            [0, 1, 2],
                            wireframe=True,
                            thicknesses=[0.6],
                        )
        dock_window(
            space=self._ui.Workspace.get_window(dock_parent_name),
            name=self._window.title,
            location=_dock_position(self._ui, dock_position),
            ratio=dock_ratio,
        )

    def _normalized_vertices(self, vertices: np.ndarray) -> np.ndarray:
        if self._center is None or self._radius is None:
            minimum = vertices.min(axis=0)
            maximum = vertices.max(axis=0)
            self._center = (minimum + maximum) / 2.0
            self._radius = max(float(np.max(maximum - minimum)) * 0.58, 1e-6)
        return (vertices - self._center) / self._radius

    @staticmethod
    def _surface_colors(vertices: np.ndarray, triangles: np.ndarray) -> list[list[float]]:
        """Return one color per indexed triangle corner, as SceneUI requires."""

        depth = vertices[:, 2]
        span = max(float(np.ptp(depth)), 1e-6)
        shade = 0.72 + 0.28 * (depth - float(depth.min())) / span
        vertex_colors = np.column_stack(
            (0.95 * shade, 0.58 * shade, 0.69 * shade, np.ones(len(vertices)))
        )
        return vertex_colors[np.asarray(triangles, dtype=np.intp).reshape(-1)].tolist()

    def update(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_update < PLOT_UPDATE_PERIOD_S:
            return
        self._last_update = now
        preview = self._session.decoded_hand()
        if preview.status.startswith("error:"):
            self._status_label.text = f"EMG2Pose {preview.status}"
        elif preview.sequence > 0:
            inference = "—" if preview.inference_ms is None else f"{preview.inference_ms:.1f} ms"
            target_hz = float(self._session.metadata.get("decoder_inference_hz_target", 5.0))
            self._status_label.text = (
                f"Decoded {self._hand} hand | target {target_hz:g} Hz | pose {preview.sequence} | "
                f"EMG sample {preview.source_sample_index} | inference {inference}"
            )
        else:
            self._status_label.text = f"EMG2Pose: {preview.status}"
        if not len(preview.vertices) or not len(preview.triangles):
            return

        topology_signature = (len(preview.vertices), len(preview.triangles))
        normalized = self._normalized_vertices(preview.vertices)
        positions = normalized.tolist()
        if topology_signature != self._topology_signature:
            vertex_counts = [3] * len(preview.triangles)
            vertex_indices = preview.triangles.reshape(-1).tolist()
            self._surface.vertex_counts = vertex_counts
            self._surface.vertex_indices = vertex_indices
            self._wireframe.vertex_counts = vertex_counts
            self._wireframe.vertex_indices = vertex_indices
            self._wireframe.colors = [[0.35, 0.10, 0.16, 0.8]] * len(vertex_indices)
            self._topology_signature = topology_signature
        if force or preview.sequence != self._displayed_sequence:
            self._surface.positions = positions
            self._surface.colors = self._surface_colors(normalized, preview.triangles)
            self._wireframe.positions = positions
            self._displayed_sequence = preview.sequence

    def close(self) -> None:
        if self._window is None:
            return
        self._window.visible = False
        self._window.destroy()
        self._window = None


def create_emg_monitor(
    session: EmgSession,
    *,
    dock_parent_name: str,
    dock_position: str = "right",
    dock_ratio: float = EMG_DOCK_RATIO,
    enabled: bool = True,
    visualize_decoder: bool = False,
    decoder_hand: str = "right",
    decoder_dock_parent_name: str | None = None,
    decoder_dock_position: str = "bottom",
    decoder_dock_ratio: float = DECODER_DOCK_RATIO,
) -> EmgMonitor | None:
    """Create a docked monitor unless Kit is headless or display is disabled."""

    from omnigibson.macros import gm

    if not enabled or gm.HEADLESS:
        return None
    return EmgMonitor(
        session,
        dock_parent_name=dock_parent_name,
        dock_position=dock_position,
        dock_ratio=dock_ratio,
        visualize_decoder=visualize_decoder,
        decoder_hand=decoder_hand,
        decoder_dock_parent_name=decoder_dock_parent_name,
        decoder_dock_position=decoder_dock_position,
        decoder_dock_ratio=decoder_dock_ratio,
    )
