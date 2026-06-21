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
  cli/              CLI commands (config, devices, joint, run)
  interface/        RobotInterface implementations
    backends/       Hardware backends (modbus_rs485)
  model/            Kinematic model (BrainCoModel)
  node/             Teleoperation control node
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
  model: revo2
  side: left
```

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

## URDF / Model Assets

URDF assets for the Revo2 kinematic model are sourced from
[BrainCoTech/brainco_hand_ros2](https://github.com/BrainCoTech/brainco_hand_ros2)
(`revo2_description` package). See that repository for license terms before
redistribution.

## Development

```bash
# Format and lint
ruff format src tests
ruff check src tests

# Run specific test file
pixi run test tests/unit/test_conversion.py
```
