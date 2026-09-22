# dex_teleop: Sharpa hand bench quick start

A right Sharpa hand fixed above a table in OmniGibson (Isaac Sim 5.1). Its 22
finger joints follow a MANUS glove (or Quest hand tracking); the wrist is held
by the simulator, so no wrist tracker is involved. Press `B` to switch between
the default view (back of the hand) and the egocentric view (palm).

These steps were verified on Ubuntu 22.04 with an RTX 4090 in a fresh conda
environment. Budget roughly 30 minutes of downloads (Isaac Sim wheels plus
3.8 GB of robot assets) and 40 GB of disk.

## 0. Prerequisites

- Ubuntu 22.04, an NVIDIA RTX GPU with about 6 GB of free VRAM, and a recent
  NVIDIA driver (the default install uses CUDA 12.8 PyTorch wheels; pass
  `--cuda-version 12.6` to `setup.sh` if the driver is older).
- `git`, `conda` (Miniconda is fine), `curl`, `unzip`.
- Two asset archives from Minyeong: `hand_bench_arat_datasets.zip` and
  `hand_bench_sharpa_robot_assets.zip` (see step 3).
- For the glove: MANUS Metagloves with their wireless dongle, the Linux SDK
  package `MANUS_Core_3.1.1_SDK.zip` (MANUS resources page or from Minyeong),
  and a `.mcal` glove calibration for the person wearing the glove.

## 1. Get the code

```bash
git clone --branch arat git@github.com:kmy17518/BEHAVIOR-1K.git Behavior-1K-arat
# or: git clone --branch arat https://github.com/kmy17518/BEHAVIOR-1K.git Behavior-1K-arat
cd Behavior-1K-arat
```

Already have a clone? `git fetch origin && git checkout arat && git pull`.
Everything below runs from this `Behavior-1K-arat` directory.

## 2. Create the conda environment

```bash
./setup.sh --new-env behavior_dex --omnigibson --bddl --dex-teleop
conda activate behavior_dex
```

Answer `y` to the Conda/NVIDIA terms prompt (or add `--accept-conda-tos
--accept-nvidia-eula`). Do **not** pass `--dataset`: it downloads the 30+ GB
BEHAVIOR-1K scene assets, which the bench does not use. The script installs
Python 3.11, PyTorch, BDDL, OmniGibson, Isaac Sim 5.1, and `dex_teleop`
(editable, with its NLopt/Pinocchio retargeting dependencies).

## 3. Assets

All assets live in `datasets/` inside the checkout (git ignores them).

```bash
# 3a. Official OmniGibson robot assets, 3.8 GB from HuggingFace -> datasets/omnigibson-robot-assets
OMNI_KIT_ACCEPT_EULA=YES python -m omnigibson.utils.asset_utils --download_omnigibson_robot_assets

# 3b. Sharpa hand + ARAT assets from the two zips (they are not downloadable)
unzip -o /path/to/hand_bench_sharpa_robot_assets.zip -d datasets/
unzip -o /path/to/hand_bench_arat_datasets.zip -d datasets/

# 3c. Check
python -c "from dex_teleop.hand_bench import load_hand_bench_scene, validate_runtime_assets; validate_runtime_assets(load_hand_bench_scene()); print('hand bench assets OK')"
```

After unzipping, `datasets/` contains:

```text
datasets/omnigibson-robot-assets/models/franka/franka.yaml                     <- replaced: adds the sharpa_right end effector
datasets/omnigibson-robot-assets/models/franka/franka_dexhand/franka_sharpa_right/  <- Franka + Sharpa USD/URDF/meshes
datasets/omnigibson-robot-assets/models/franka_mounted_sharpa_{right,left}/     <- standalone Franka+Sharpa robot models
datasets/arat-assets-v1/                                                        <- table used by the bench, ARAT objects, arat_base scene
datasets/arat-task-instances/                                                   <- 19 ARAT task scene instances (ARAT launcher only)
```

## 4. Smoke test without hardware

```bash
# Interactive window: the hand stays open; B toggles the view, Ctrl+C exits
python dex_teleop/scripts/launch_hand_bench.py --view-only

# No display (e.g. over SSH): render both views to PNG and exit
OMNIGIBSON_HEADLESS=1 python dex_teleop/scripts/launch_hand_bench.py --screenshot-dir dex_teleop/outputs/hand_bench
```

The first launch compiles shaders and can take several minutes; later launches
take about 30 s. A successful load prints `Hand bench ready: right_hand_C_MC at
[0.0, 0.0, 0.891] ... tabletop at 0.750 m`.

## 5. MANUS Integrated SDK (Linux, gloves only)

The bench talks to the gloves through a small C++ sidecar that is compiled
against the official SDK; the proprietary SDK files stay outside the repository.

```bash
# 5a. Build tools and libraries the SDK needs
sudo apt install build-essential libusb-1.0-0-dev libudev-dev zlib1g-dev

# 5b. Unpack the SDK where dex_teleop looks for it (the zip already contains a
#     top-level ManusSDK_v3.1.1/ folder); MANUS_SDK_ROOT=<dir> overrides the search
mkdir -p ~/manus_setup/vendor
unzip MANUS_Core_3.1.1_SDK.zip -d ~/manus_setup/vendor/
ls ~/manus_setup/vendor/ManusSDK_v3.1.1/SDKClient_Linux/ManusSDK/{include,lib}
#   must show ManusSDK.h ... and libManusSDK_Integrated.so

# 5c. Let your user open the dongle (once), then re-plug the dongle
sudo tee /etc/udev/rules.d/70-manus-hid.rules >/dev/null <<'EOF'
# HIDAPI/libusb
SUBSYSTEMS=="usb", ATTRS{idVendor}=="3325", MODE:="0666"
SUBSYSTEMS=="usb", ATTRS{idVendor}=="1915", ATTRS{idProduct}=="83fd", MODE:="0666"

# HIDAPI/hidraw
KERNEL=="hidraw*", ATTRS{idVendor}=="3325", MODE:="0666"
EOF
sudo udevadm control --reload-rules && sudo udevadm trigger

# 5d. Build the Integrated sidecar (-> ~/.cache/dex_teleop/manus_bridge/manus_bridge)
bash dex_teleop/src/dex_teleop/native/manus_bridge/build_manus_bridge.sh --mode integrated --sdk-root ~/manus_setup

# 5e. Validate with the glove on, dongle plugged in; move your fingers for 10 s
dex-teleop-manus-diagnostics --mode integrated --hand right --duration 10 \
  --glove-calibration /path/to/right-hand.mcal --output /tmp/manus_diagnostics.jsonl
```

The diagnostics print a JSON summary and exit 0 when `"passed": true`. If the
build already exists it is reused; rerun 5d after updating the SDK. Pairing,
firmware updates, license provisioning, and creating the `.mcal` calibration
are done in MANUS Core on Windows; the Linux Integrated SDK only consumes
them. The `.mcal` is optional (`--manus-calibration` may be omitted) but finger
angles are only accurate with the wearer's own calibration.

## 6. Run the hand bench with the glove

```bash
python dex_teleop/scripts/launch_hand_bench.py \
  --hand-source manus --manus-calibration /path/to/right-hand.mcal
```

The fingers follow the glove as soon as the first frame arrives. Keys: `B`
toggles default/egocentric view, `SPACE` pauses and resumes, `R` reopens the
hand, Ctrl+C exits. Useful options: `--stale-frame-policy hold` (hold the last
pose instead of failing on a stale stream), `--finger-target-scale 0.8`
(shrink flexion), `--retargeter dexpilot` (optional alternative backend,
`pip install -e './dex_teleop[dexpilot]'`), `--show-arm` (reveal the hidden
Franka that carries the hand). Wrist options (`--wrist-source`, `--source`,
VIVE/tracker flags) are rejected on purpose. Quest hand tracking works too:
`--hand-source quest` listens for the HTS app on UDP 9000.

## 7. Tests

```bash
pytest -q dex_teleop/tests
```

## Troubleshooting

- `Out of GPU memory` during startup: another process holds the VRAM; free
  about 6 GB.
- `OMNIGIBSON_DATA_PATH points outside Behavior-1K-arat`: unset that variable;
  the launchers always use the checkout's `datasets/`.
- `Missing required hand bench assets`: step 3 is incomplete; the message
  lists the missing files.
- `Could not find a matching MANUS SDK header and libManusSDK_Integrated.so`:
  check the path in 5b or set `MANUS_SDK_ROOT=~/manus_setup/vendor/ManusSDK_v3.1.1`.
- `MANUS bridge did not publish a validated right-hand articulation within
  N seconds` (diagnostics or the bench): the SDK and sidecar are fine but no
  glove data arrived. Power the glove, re-plug the dongle, check the udev rule
  (5c), and confirm in MANUS Core (Windows) that the dongle carries the SDK
  license.

## More

The ARAT task launcher (`dex_teleop/scripts/launch_og.py`), recording format,
replay, Quest/VIVE wrist tracking, MANUS Remote mode, and EMG recording are
documented in the previous README, kept in `dex_teleop/docs/` (untracked; ask
Minyeong) and in git history (`git show a8ddd6bfd:dex_teleop/README.md`).
