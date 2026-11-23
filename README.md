# BrainCo Revo2 Dexterous Hand Control

> [!IMPORTANT]
> **This repository is archived.** `dexim-brainco` now lives in the dexim
> monorepo at
> [`packages/dexim-brainco`](https://github.com/TechShare-inc/dexim/tree/dev/packages/dexim-brainco),
> imported from `775ae03` of this repository. Open issues and pull requests
> there. The content below describes the last state of this repository.

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

## Revo2 / Revo3 offline support

This checkout adds explicit `revo2` and `revo3`, left/right, synthetic/replay
support. It does **not** qualify physical hardware. `run` is the legacy Revo2
node; a Revo3 config is rejected there. Hardware connection, scanning and
calibration are gated pending a qualified device profile. Install this unpublished
candidate from its explicit local path; do not advance umbrella gitlinks.

```bash
pip install -e /tmp/ts-dexim-brainco-package --no-deps
dexim-brainco offline --model revo2 --side left --check
dexim-brainco offline --model revo3 --side right --steps 20
# JSON array of core HandState.to_dict() records:
dexim-brainco offline --model revo3 --side left --steps 20 --replay hand.json
```

The command is bounded to 1–1000 steps, emits JSON lines, and never loads a
native SDK. Each line separates `action` (successful issued write) from
`observation` (independent mock read). A replay retains original capture times
in `replay.original_timestamp`, publishes fresh Unix capture time, preserves
record order and stops at EOF. The standalone demonstration uses deterministic
20 ms control ticks without sleeping; the umbrella worker owns real-time replay
scheduling. It is not a real-time hardware benchmark.

The public import is `dexim.brainco.offline`: `OfflineConfig`,
`describe_model`, `synthetic_hand`, `load_replay`, `OfflineController` and the
immutable action/observation/report records. Config defaults retain Revo2;
new callers should explicitly name model and side. Unknown config fields,
malformed/duplicate IDs, wrong glove/side, invalid source timing, nonfinite
positions and invalid quaternions are rejected. The explicit 25-point layout is
root, thumb (4 points), then index/middle/ring/little (5 points each). Custom
ordered IDs are allowed; their anatomical ordering must match this layout.

```python
import time
from dexim.brainco.offline import OfflineConfig, OfflineController, synthetic_hand

config = OfflineConfig(model="revo3", side="left", glove_id=1)
controller = OfflineController(config)
controller.prepare()  # observations available before activation
controller.activate("unique-epoch", time.monotonic() + 0.02)
# The worker calls watchdog(epoch, strictly_increasing_sequence) and step(...)
# at its bounded cadence. Capture timestamps are Unix; deadlines are monotonic.
result = controller.step(synthetic_hand(config, time.time()), epoch_id="unique-epoch")
controller.end("unique-epoch")  # observations remain available
controller.shutdown()
```

Watchdog, input loss, stale observations, invalid solves and failed writes latch
faults and revoke authority. Epoch IDs cannot be reused; old work cannot write
in a later epoch. A fault requires a new controller instance. All safety reports
are software/mock outcomes, never evidence of physical stop. Maintenance and
prepared session ownership exclude each other.

### Model facts and retargeting limits

Descriptors use named canonical URDF independent order and radians, separate
from SDK order. Revo2 has 6 independent/11 full joints; five mimic positions are
explicitly **derived**, not independently measured. New Revo2 limits use the
canonical rounded radians (1.57/1.03/1.41), preserving legacy conversion's
90/59/81-degree constants only in the explicit legacy path. Revo3 has 21
independent joints with distinct left/right bounds. The URDF's zero effort and
velocity metadata are not runtime limits; `max_velocity` is an explicit offline
limit in radians/second (default 2). Mock torque/velocity feedback is unavailable
(NaN in core JointState), and currents are never labeled torque.

Retargeting uses root-quaternion hand-local geometry: Revo2 deliberately reduces
finger curl into coupled axes; Revo3 retains MCP palm-plane flexion, PIP/DIP
segment bends, finger spread and bounded thumb opposition. Thumb's four source
points cannot fully identify all five independent thumb axes. This geometric
approximation is deterministic and bounded, not native IK or physical tracking
acceptance. Filtering is followed by final position and velocity clamps.

`joints.json` contains numeric facts from official description revision
`324cd0c5fcf71e077d8daa9514ef259ade85eba7`; `asset_hashes.json` records the four
canonical URDF SHA256 hashes. No external URDF, mesh or SDK binary is included.
`descriptor.validate_assets(path)` requires that exact model/side hash and
available meshes; missing/unlicensed assets must be supplied separately. Revo3
SDK thumb aliases follow the pinned joint-state aligner/SDK table order
MCP/PIP/DIP/CMP/CMR; physical signs, offsets and limit readback remain unqualified.

### Observed-state rendering

Rendering is package-local and opt-in. Ordinary package/CLI/offline imports do
not import Pinocchio, Viser, trimesh or vendor SDKs. With optional native FK and
Viser installed and a licensed canonical external URDF supplied:

```python
from dexim.brainco.offline import describe_model
from dexim.brainco.offline.render import ObservedRenderer

renderer = ObservedRenderer(describe_model("revo3", "left"),
                            "/external/revo3_left.urdf", server)
renderer.update(result.observation)
renderer.close()
```

This displays the FK joint skeleton, including derived Revo2 mimic expansion,
and rejects action records or mismatched model/side/names. It does not claim
mesh rendering. FK requires the pinned external URDF but does not need meshes;
mesh validation is separately available. The old session renderer can be
registered explicitly with `dexim.brainco.register_visualizer_renderer()` when
using the legacy visualizer. New offline observations are rendered through the
typed renderer above, not the legacy commanded-action subscriber.

### SDK seams and verification

Separate Revo2 (`bc-stark-sdk==2.0.9`) and Revo3 (`bc-revo3-sdk==2.3.5`) fake
seams exercise pinned stub/example APIs. SDK dependencies remain optional;
physical activation without qualification is rejected before native imports.
Revo2 selects `FingerUnitMode.Normalized`, never calibration. Revo3 uses one-shot
`motion.set_position`, not scheduled `move_to`. Awaitables are bounded; errors
propagate without cached healthy state. A bus has one owner. A completed write,
timeout, cancellation or close is not mechanical stop evidence. Injected SDKs
are a caller trust boundary and must be fakes in offline work.

`pixi run --locked test` is the mandatory task; this unchanged lock supports
only win-64, so Linux reports unsupported-platform. Additional diagnostics use
the existing locked Linux umbrella Python with verified pinned core sources:

```bash
PYTHONPATH=src:/tmp/ts-dexim-dev-verify-20261113/packages/dexim-core/src \
  /tmp/ts-dexim-dev-verify-20261113/.pixi/envs/default/bin/python \
  -m pytest tests/unit tests/mock -m 'not hardware and not requires_model'
```

No hardware tests, device scans, connections, calibration or motion form part of
this offline acceptance. Hardware identity, firmware, transports, signs/zeros,
limits, health semantics and confirmed stop behavior require separately
authorized qualification. Live MANUS/capture migration remains separate.
