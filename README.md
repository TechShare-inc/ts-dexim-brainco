# BrainCo Revo2 Dexterous Hand Control

`dexim-brainco` is a typed Python package for BrainCo Revo2 dexterous hand
control over RS-485 / Modbus, built on the `dexim-core` teleoperation framework.

## Quick Start

```bash
# Install dev dependencies
pixi install

# Run tests (no hardware required)
pixi run test

# Run tests with coverage
pixi run test-cov

# Install hardware support
pixi run pip install bc-stark-sdk
```

## Package Layout

```
src/dexim/brainco/
  cli/              CLI commands (run, status, probe, check, config, model, joint, devices)
  interface/        RobotInterface implementations
    backends/       Hardware backends (modbus_rs485)
  model/            Kinematic model (BrainCoModel)
  node/             Control node (extractor, retargeter, filter, vel-limit, publish)
  viz/              3D visualization (Viser-based renderer + ZMQ subscriber)
```

## CLI Usage

```bash
# Start the control node (mock mode)
 dexim-brainco run --mode mock --hand left

# Start with hardware (Modbus RS-485)
 dexim-brainco run --mode hw --port COM5 --baud 460800 --hand left

# Use a named device from config
 dexim-brainco run --device left_revo2

# Manage configs
 dexim-brainco config new lab-revo2 --hand left
 dexim-brainco config list

# Register named devices
 dexim-brainco devices add left_revo2 --config lab-revo2
 dexim-brainco devices list

# Direct joint commands (hardware required)
 dexim-brainco joint open --port COM5
 dexim-brainco joint close --port COM5
 dexim-brainco joint watch --port COM5

# Launch 3D visualizer
 dexim-brainco viz --hand left --port 8080

# Check device connectivity
 dexim-brainco probe --port COM5
 dexim-brainco check --port COM5
```

## Configuration

Device configs are stored under `{config_dir}/dexim/`:

- `{config_dir}/dexim/devices.yaml` — device registry
- `{config_dir}/dexim/brainco/<name>.yaml` — per-device config

Example `devices.yaml`:

```yaml
left_revo2:
  package: dexim-brainco
  config: left-revo2
```

Example `left-revo2.yaml`:

```yaml
interface:
  mode: hw
  hw:
    backend: modbus_rs485
    rs485:
      port: COM5
      baud: 460800
    slave_id: 126
    auto_detect: false
    auto_calibrate: false
    hardware_core:
      rate_hz: 30.0
      shm_prefix: brainco
      carrot_lookahead_cycles: 0.0

brainco:
  side: left

control:
  rate_hz: 30.0
  enable_velocity_limiting: true
  max_joint_velocity_rad_s: 1.0

subscriber:
  address: tcp://localhost:5555
```

### Control Pipeline

The node runs the following pipeline each tick (30 Hz):

1. **Receive** skeleton data from the Manus glove subscriber
2. **Extract** finger-direction vectors from joint positions
3. **Scale** feature vectors (per-finger alpha)
4. **Retarget** using direct linear mapping (finger vectors → joint angles)
5. **Filter** joint angles (WMA/EMA/One Euro)
6. **Velocity-limit** joint commands for safety
7. **Send** joint targets to the hardware via shared memory
8. **Publish** action and observation data for visualization

Set `DEXIM_CONFIG_DIR` to override the config directory.

## RS-485 / Modbus

- Default slave IDs: left `126` (`0x7E`), right `127` (`0x7F`)
- Default baud rate: `460800`
- SDK position convention: `0` = open, `1000` = closed
- 6 active joints: thumb flex, thumb aux, index, middle, ring, pinky

## Safety

- Hardware access is **isolated in a child process**. Only the Hardware Core
  process touches `bc_stark_sdk` and the RS-485 serial port.
- `auto_calibrate` is **disabled by default**. Enable only after confirming
  hardware safety and that the hand has completed power-on calibration.
- The hand starts at open position (lower joint limits = 0 rad).

## Hardware Tests

Hardware tests are opt-in and excluded from default test runs:

```bash
pixi run test-hardware
```

Requires a connected BrainCo Revo2 hand.

## Vendored Assets

The Revo2 URDF models and 3D meshes are vendored from
[BrainCoTech/revo2_description](https://github.com/BrainCoTech/revo2_description)
under the `vendors/` directory. This allows the kinematic model to load URDF
files without requiring ROS 2 to be installed.

### Structure

```
vendors/
└── revo2_description/        # git clone of BrainCoTech/revo2_description
    ├── urdf/                 # .urdf (and .urdf.xacro) for left/right hands
    ├── meshes/               # .STL mesh files
    ├── launch/               # ROS 2 launch files (unused by dexim-brainco)
    └── rviz/                 # RViz config files (unused by dexim-brainco)
```

### Setup

The vendor is **tracked as a full Git clone** inside this repository.
After cloning `ts-dexim-brainco`, initialise the vendor:

```bash
cd ts-dexim-brainco
git submodule update --init --recursive
```

> If `vendors/revo2_description` is still empty, clone it manually:
>
> ```bash
> git clone https://github.com/BrainCoTech/revo2_description.git vendors/revo2_description
> ```

### Updating the Vendor

To pull the latest URDF updates from upstream:

```bash
cd vendors/revo2_description
git pull origin main
```

Then commit the updated vendor in the parent repo:

```bash
cd ../..
git add vendors/revo2_description
git commit -m "chore(vendor): update revo2_description from upstream"
```

### How the Model Uses It

`BrainCoModel` (in `src/dexim/brainco/model/model.py`) automatically resolves
the URDF path relative to the vendored directory and builds a Pinocchio
kinematic model at runtime:

```python
from dexim.brainco.model.factory import create_model

# Left hand (default)
left_model = create_model("left")
print(left_model.nq, left_model.tip_frame_names)

# Right hand
right_model = create_model("right")
```

The model sets `ROS_PACKAGE_PATH` temporarily during build so Pinocchio can
resolve `package://revo2_description/...` mesh references inside the URDF.

### License

The vendored assets are from [BrainCoTech/revo2_description](https://github.com/BrainCoTech/revo2_description),
licensed under **Apache 2.0**. See `vendors/revo2_description/LICENSE` for the
full license text.

## Development

```bash
# Format and lint
ruff format src tests
ruff check src tests

# Run specific test file
pixi run test tests/unit/test_conversion.py
```
