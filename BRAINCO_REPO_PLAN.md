# ts-dexim-brainco Repository Plan

Date: 2026-06-21

## Goal

Create a new repository named `ts-dexim-brainco` for BrainCo Revo2 dexterous hand control, using this `ts-dexim-inspire` repository as the structural sample project.

The target implementation should control BrainCo Revo2 over RS-485 / Modbus using Python and BrainCo's current SDK.

## Existing Sample Project: ts-dexim-inspire

The current repository is a typed Python package for dexterous hand control:

```text
ts-dexim-inspire/
  pyproject.toml
  pixi.toml
  README.md
  src/dexim/inspire/
    cli/
    interface/
    model/
    node/
    viz/
  tests/
    unit/
    mock/
    hardware/
```

Important design points to preserve:

- Public package namespace: `dexim.<hand_vendor>`.
- CLI entrypoint for direct hardware/config/joint operations.
- YAML-based config and device registry.
- Mock, unit, and hardware test separation.
- `RobotInterface` abstraction from `dexim-core`.
- Shared-memory parent/child architecture for hardware control.
- Hardware access isolated in a child process.
- Main control node remains focused on skeleton receive, feature extraction, retargeting, filtering, velocity limiting, command send, and observation publish.

The most important architectural pattern is:

```text
Brain/Gateway process
  -> RobotInterface.write()
  -> shared memory target buffer
  -> Hardware Core child process
  -> hardware SDK / transport
  -> shared memory state buffer
  -> RobotInterface.read()
```

This pattern should be kept for BrainCo because SDK/serial timing and blocking I/O should not run inside the high-level teleoperation loop.

## BrainCo SDK Findings

Sources checked:

- BrainCo SDK repository: https://github.com/BrainCoTech/brainco-hand-sdk
- Revo2 product/docs page: https://www.brainco-hz.com/docs/revolimb-hand/revo2/parameters.html
- Revo2 Python SDK docs: https://www.brainco-hz.com/docs/revolimb-hand/revo2/python_sdk.html
- Python demo examples: https://github.com/BrainCoTech/brainco-hand-sdk/tree/main/python/demo
- Revo2 RS-485 examples: https://github.com/BrainCoTech/brainco-hand-sdk/tree/main/python/revo2

Current public SDK facts:

- Python SDK package is `bc_stark_sdk`.
- SDK examples import `from bc_stark_sdk import main_mod as sdk`.
- Revo2 supports RS-485, CANFD, and EtherCAT.
- The requested initial target should use RS-485 / Modbus.
- Revo2 has 6 active joints and 11 total DOF.
- Default Revo2 IDs:
  - left hand: `126` / `0x7E`
  - right hand: `127` / `0x7F`
- Python supported versions in BrainCo docs/examples: Python 3.9 to 3.12.
- Typical examples use async APIs.
- Relevant SDK functions/classes include:
  - `sdk.auto_detect_modbus_revo2(...)`
  - `sdk.modbus_open(...)`
  - `sdk.modbus_close(...)`
  - `DeviceContext.get_device_info(...)`
  - `DeviceContext.set_hardware_type(...)`
  - `DeviceContext.set_finger_unit_mode(...)`
  - `DeviceContext.set_finger_positions(...)`
  - `DeviceContext.set_finger_positions_and_durations(...)`
  - `DeviceContext.set_finger_positions_and_speeds(...)`
  - `DeviceContext.get_motor_status(...)`

BrainCo normalized position convention from examples:

```text
0    = open
1000 = closed
```

Revo2 SDK/demo finger command order:

```text
[Thumb, ThumbAux, Index, Middle, Ring, Pinky]
```

Revo2 documented active joint angle ranges:

```text
Thumb flex: 0 to 59 deg
Thumb aux:  0 to 90 deg
Index:      0 to 81 deg
Middle:     0 to 81 deg
Ring:       0 to 81 deg
Pinky:      0 to 81 deg
```

## Proposed Repository Structure

```text
ts-dexim-brainco/
  pyproject.toml
  pixi.toml
  README.md
  src/dexim/brainco/
    __init__.py
    py.typed
    cli/
      __init__.py
      commands.py
      config_utils.py
      devices.py
      joint.py
    interface/
      __init__.py
      py.typed
      brainco_interface.py
      hardware_core.py
      config.py
      factory.py
      mock_interface.py
      _conversion.py
      _process_logger.py
      backends/
        __init__.py
        modbus_rs485.py
    model/
      __init__.py
      py.typed
      model.py
      factory.py
      assets/
    node/
      __init__.py
      py.typed
      node.py
      config.py
      extractor.py
      retargeter.py
    viz/
      __init__.py
      renderer.py
      subscriber.py
  tests/
    conftest.py
    unit/
    mock/
    hardware/
```

## Package Metadata

`pyproject.toml` should change from Inspire-specific metadata to BrainCo-specific metadata:

```toml
[project]
name = "dexim-brainco"
description = "BrainCo Revo2 dexterous hand model, interface, and teleoperation control node"
requires-python = ">=3.10"

[project.scripts]
dexim-brainco = "dexim.brainco.cli:standalone_app"
```

Expected dependencies:

```toml
dependencies = [
    "numpy",
    "loguru",
    "dexim-core",
    "dexim-cli-common",
    "rich-click",
]

[project.optional-dependencies]
hardware = [
    "bc-stark-sdk",
]
viz = [
    "pinocchio",
    "viser",
    "trimesh",
    "scipy",
]
dev = [
    "pytest",
    "pytest-cov",
    "ruff",
]
```

Note: BrainCo's current `python/requirements.txt` pins `numpy<2.0.0` because of GUI/PySide constraints. If this project does not include PySide GUI, validate whether `bc-stark-sdk` itself works with the existing project's `numpy>=2.2.0`. If not, pin `numpy<2`.

## Configuration Design

Keep the same device registry pattern:

```text
{config_dir}/dexim/devices.yaml
{config_dir}/dexim/brainco/<name>.yaml
```

Example device registry:

```yaml
left_revo2:
  package: dexim-brainco
  config: left-revo2

right_revo2:
  package: dexim-brainco
  config: right-revo2
```

Example Revo2 RS-485 config:

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
  model: revo2
  side: left
  glove_id: null
```

Config fields to support:

- `mode`: `mock` or `hw`
- `backend`: initially `modbus_rs485`
- `rs485.port`: serial port, optional if auto-detect is enabled
- `rs485.baud`: default configurable, likely `460800`
- `slave_id`: default from side if omitted
- `auto_detect`: whether to use `sdk.auto_detect_modbus_revo2`
- `auto_calibrate`: whether to call calibration on connect
- `hardware_core.rate_hz`
- `hardware_core.shm_prefix`
- `hardware_core.carrot_lookahead_cycles`

## Interface Design

### Public Interface

Create `BrainCoInterface`, equivalent in role to `InspireInterface`:

- Implements `dexim.core.robot_interface.RobotInterface`.
- Creates target/state shared memory on `connect()`.
- Spawns `run_hardware_core(...)` in a child process.
- `write()` writes target joint positions to shared memory.
- `read()` reads latest device state from shared memory.
- `disconnect()` stops child process and unlinks shared memory.

### Hardware Core

Create `dexim.brainco.interface.hardware_core.run_hardware_core`.

It should preserve the existing loop shape:

1. Read target joint angles from shared memory.
2. Optionally apply target extrapolation.
3. Send target to BrainCo backend.
4. Read motor status from BrainCo backend.
5. Write state into shared memory.
6. Sleep with `RateLimiter`.

The hardware core process is the only process that imports and touches `bc_stark_sdk`.

## RS-485 Backend Design

Create `BrainCoModbusRS485Backend`.

Responsibilities:

- Wrap BrainCo SDK async APIs behind a synchronous `RobotInterface` boundary.
- Own the SDK `DeviceContext`.
- Connect using either explicit serial settings or auto-detection.
- Convert internal radians to BrainCo normalized `0..1000`.
- Convert BrainCo motor status positions to internal radians.
- Handle read fallback to last known position.
- Expose joint metadata.

Because BrainCo SDK APIs are async, the backend should own an event loop inside the hardware-core process.

Implementation options:

1. Simple phase 1: use `asyncio.run(...)` per call. Easier but may add overhead.
2. Preferred phase 2: create a persistent event loop in the backend and run coroutines with `loop.run_until_complete(...)`.

The preferred backend shape:

```python
class BrainCoModbusRS485Backend(RobotInterface):
    def connect(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._connect_async())

    def write(self, cmd: JointCommand) -> None:
        positions = radians_to_api(cmd.q)
        self._loop.run_until_complete(
            self._client.set_finger_positions(self._slave_id, positions.tolist())
        )

    def read(self) -> JointState:
        status = self._loop.run_until_complete(
            self._client.get_motor_status(self._slave_id)
        )
        q = api_to_radians(status.positions)
        return JointState(q=q, qd=zeros, tau=zeros, stamp=time.time())
```

## Conversion Design

Internal order should be chosen to match the hand model and retargeter. A practical starting point is to use BrainCo device order as internal active order until URDF/model details require otherwise:

```text
Internal active order:
[thumb_flex, thumb_aux, index, middle, ring, pinky]

BrainCo SDK order:
[thumb_flex, thumb_aux, index, middle, ring, pinky]
```

If the model/URDF later expects a different order, keep the conversion explicit with maps:

```python
INTERNAL_TO_API_MAP = [...]
API_TO_INTERNAL_MAP = [...]
```

Use documented limits:

```python
JOINT_LIMITS = [
    (0.0, radians(59.0)),  # thumb flex
    (0.0, radians(90.0)),  # thumb aux
    (0.0, radians(81.0)),  # index
    (0.0, radians(81.0)),  # middle
    (0.0, radians(81.0)),  # ring
    (0.0, radians(81.0)),  # pinky
]
```

Conversion:

```text
api = ((q - q_min) / (q_max - q_min)) * 1000
q   = (api / 1000) * (q_max - q_min) + q_min
```

Unlike Inspire, do not invert the range because BrainCo examples use `0=open`, `1000=closed`, which naturally maps lower limit to open and upper limit to closed.

## CLI Plan

Keep the same command families as `dexim-inspire`:

```bash
dexim-brainco devices scan
dexim-brainco config new left-revo2 --mode hw --port COM5 --baud 460800 --side left
dexim-brainco devices add left_revo2 --config left-revo2
dexim-brainco probe --device left_revo2
dexim-brainco check --device left_revo2
dexim-brainco joint open --device left_revo2
dexim-brainco joint close --device left_revo2
dexim-brainco joint home --device left_revo2
dexim-brainco joint set 0 0 0 0 0 0 --device left_revo2
dexim-brainco joint watch --device left_revo2
dexim-brainco run --device left_revo2
```

BrainCo-specific options:

```bash
--slave-id 126
--auto-detect
--auto-calibrate
--baud 460800
```

## Test Plan

Unit tests:

- Conversion:
  - radians to API
  - API to radians
  - clipping
  - round-trip conversion
  - joint order mapping
- Config:
  - default left/right slave IDs
  - explicit slave ID override
  - RS-485 config validation
  - auto-detect config validation
- Mock interface:
  - write/read loopback
  - joint metadata
- Factory:
  - mock construction
  - hardware construction config translation

Hardware tests, marked `hardware`:

- `devices scan`
- explicit connect
- auto-detect connect
- read `get_device_info`
- read motor status
- open hand
- close hand
- home/safe pose
- fixed-rate watch for N samples

## Implementation Phases

### Phase 1: Scaffold

- Copy project structure from `ts-dexim-inspire`.
- Rename package/imports:
  - `dexim.inspire` -> `dexim.brainco`
  - `Inspire` -> `BrainCo` or `BrainCoRevo2`
  - `dexim-inspire` -> `dexim-brainco`
- Remove Inspire-specific DDS/TCP/IP paths initially.
- Keep mock, CLI, config, tests.

### Phase 2: Conversion and Metadata

- Implement Revo2 joint names.
- Implement documented Revo2 joint limits.
- Implement API/radian conversion.
- Add conversion unit tests.

### Phase 3: RS-485 Backend

- Add `bc-stark-sdk` optional hardware dependency.
- Implement `BrainCoModbusRS485Backend`.
- Use explicit port/baud/slave ID first.
- Add auto-detect after explicit connection works.
- Add safe read/write fallbacks.

### Phase 4: Hardware Core Integration

- Wire backend factory into `BrainCoInterface`.
- Verify shared-memory hardware loop works.
- Add hardware smoke tests.

### Phase 5: CLI Bring-Up

- Implement `devices scan` using BrainCo SDK port listing and/or auto-detect.
- Implement `probe`, `check`, `joint open`, `joint close`, `joint set`, `joint watch`.
- Document first-time hardware workflow.

### Phase 6: Model and Retargeting

- Add BrainCo Revo2 URDF/assets.
- Implement `BrainCoModel`.
- Verify active joint order matches backend conversion.
- Reuse existing feature extraction/retargeting pattern.

### Phase 7: Optional Extensions

- CANFD backend.
- EtherCAT backend.
- Tactile telemetry support.
- Action sequence support.
- Higher-frequency SDK data collector integration if needed.

## Main Risks

- `bc-stark-sdk` async methods may not tolerate repeated event loop creation. Prefer a persistent event loop inside the backend.
- SDK/numpy compatibility must be verified. BrainCo examples currently constrain `numpy<2.0.0` for GUI reasons, but the core SDK may or may not require it.
- Revo2 model/URDF joint order may differ from SDK order. Keep conversion maps explicit.
- Calibration behavior matters. BrainCo docs say the hand must complete position calibration after power-on before normal control. Do not auto-calibrate by default until hardware safety is confirmed.
- Thumb has two active joints and examples note special behavior in some trajectory helpers. Treat thumb flex and thumb aux explicitly.

## Recommended First Commit Scope

The first implementation commit should only include:

- package scaffold
- config dataclasses
- conversion module
- mock interface
- backend skeleton with SDK import guarded behind hardware extra
- unit tests for conversion/config/mock
- README quick start draft

Do not include CANFD, EtherCAT, tactile handling, or full URDF retargeting in the first commit.
