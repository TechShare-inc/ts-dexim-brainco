"""Offline safety/model contracts without native SDK or device I/O."""

import json
from unittest.mock import Mock

import numpy as np
import pytest

from dexim.brainco.offline import (
    OfflineConfig,
    OfflineController,
    describe_model,
    load_replay,
    synthetic_hand,
)
from dexim.brainco.offline.hand import normalize_hand, retarget_hand


@pytest.mark.parametrize("model,n,full", [("revo2", 6, 11), ("revo3", 21, 21)])
@pytest.mark.parametrize("side", ["left", "right"])
def test_models_mapping_and_poses(model, n, full, side):
    cfg = OfflineConfig(model=model, side=side)
    d = describe_model(model, side)
    assert len(d.joint_names) == n and len(d.full_joint_names) == full
    q = np.array(d.lower) * 0.3 + np.array(d.upper) * 0.7
    np.testing.assert_allclose(d.from_sdk(d.to_sdk(q)), q, atol=1e-12)
    assert len(d.expand(q)) == full
    outputs = [
        retarget_hand(synthetic_hand(cfg, 100.0, p), cfg)
        for p in ("open", "fist", "pinch", "spread", "opposition")
    ]
    for output in outputs:
        assert np.isfinite(output).all()
        assert (output >= d.lower).all() and (output <= d.upper).all()
    assert np.max(abs(outputs[1] - outputs[0])) > 0.5
    assert np.max(abs(outputs[2] - outputs[0])) > 0.5
    assert np.max(abs(outputs[4] - outputs[0])) > 0.1
    if model == "revo3":
        assert np.max(abs(outputs[3] - outputs[0])) > 0.1


@pytest.mark.parametrize("model", ["revo2", "revo3"])
def test_wrist_and_side_invariance(model):
    cfg = OfflineConfig(model=model)
    hand = synthetic_hand(cfg, 100.0, "fist")
    original = retarget_hand(hand, cfg)
    for j in hand.joints:
        x, y, z = j.position
        j.position = (-y + 1, x + 2, z + 3)
        j.rotation = (2**-0.5, 0.0, 0.0, 2**-0.5)
    np.testing.assert_allclose(retarget_hand(hand, cfg), original, atol=1e-7)
    right = OfflineConfig(model=model, side="right")
    np.testing.assert_allclose(
        normalize_hand(synthetic_hand(right, 100.0, "fist"), right),
        normalize_hand(synthetic_hand(cfg, 100.0, "fist"), cfg),
    )


def test_revo3_segment_flexion_is_independent():
    cfg = OfflineConfig(model="revo3")
    d = describe_model("revo3")

    def bend(at):
        hand = synthetic_hand(cfg, 100.0)
        pivot = np.array(hand.joints[at].position)
        for j in hand.joints[at + 1 : 10]:
            offset = np.array(j.position) - pivot
            j.position = tuple(pivot + [offset[0], 0.0, -offset[1]])
        return retarget_hand(hand, cfg)

    pip, dip = bend(7), bend(8)
    assert pip[d.joint_names.index("left_index_PIP_joint")] > 1
    assert dip[d.joint_names.index("left_index_PIP_joint")] < 1e-6
    assert dip[d.joint_names.index("left_index_DIP_joint")] > 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("model", "other"),
        ("side", "bad"),
        ("mode", "hw"),
        ("max_velocity", float("nan")),
        ("glove_id", True),
        ("joint_ids", tuple([0] * 25)),
    ],
)
def test_config_rejects(field, value):
    with pytest.raises((ValueError, TypeError)):
        OfflineConfig(**{field: value})


def test_replay_validates_whole_file(tmp_path):
    cfg = OfflineConfig()
    path = tmp_path / "replay.json"
    rows = [synthetic_hand(cfg, t).to_dict() for t in (100.0, 101.0)]
    path.write_text(json.dumps(rows))
    assert len(load_replay(path, cfg)) == 2
    for bad in ({}, [rows[0], rows[0]], [{"side": "left"}]):
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            load_replay(path, cfg)
    rows[0]["joints"][1]["id"] = True
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError):
        load_replay(path, cfg)


def controller():
    mono, wall = [0.0], [100.0]
    c = OfflineController(
        OfflineConfig(), clock=lambda: mono[0], wall_clock=lambda: wall[0]
    )
    assert c.prepare().outcome == "prepared"
    assert c.step().observation is not None
    assert c.activate("a", 0.1).outcome == "scheduled"
    return c, mono, wall


def test_epoch_streams_and_old_work():
    c, mono, wall = controller()
    hand = synthetic_hand(c.config, wall[0], "fist")
    assert c.step(hand, epoch_id="a").action is None
    mono[0] = 0.11
    assert c.step(hand, epoch_id="a").action is not None
    c.end("a")
    assert c.step().observation is not None
    assert c.activate("a", 0.2).outcome == "rejected"
    assert c.activate("b", 0.2).outcome == "scheduled"
    assert c.end("a").outcome == "rejected"
    assert c.active_epoch == "b"
    assert c.watchdog("a", 100).outcome == "rejected"
    mono[0] = 0.21
    assert c.step(hand, epoch_id="a").action is None
    assert c.active_epoch == "b"


@pytest.mark.parametrize(
    "failure",
    ["watchdog", "input", "read", "write", "stale", "duplicate", "future", "deadline"],
)
def test_failures_latch_and_revoke(failure, monkeypatch):
    c, mono, wall = controller()
    mono[0] = 0.11
    hand = synthetic_hand(c.config, wall[0], "fist")
    if failure == "watchdog":
        c.watchdog("a", 1)
        mono[0] = 0.7
        c.watchdog("a", 1)
    elif failure == "input":
        c.step()
        mono[0] = 0.4
    elif failure == "read":
        c.interface.read = Mock(side_effect=RuntimeError("read failed"))
    elif failure == "write":
        c.interface.write = Mock(side_effect=RuntimeError("write failed"))
    elif failure == "stale":
        hand.timestamp -= 1
    elif failure == "future":
        hand.timestamp += 1
    elif failure == "duplicate":
        assert c.step(hand, epoch_id="a").action
    elif failure == "deadline":
        from dexim.brainco.offline import controller as module

        original = module.retarget_hand

        def delayed(*args):
            result = original(*args)
            mono[0] += 0.1
            return result

        monkeypatch.setattr(module, "retarget_hand", delayed)
    result = c.step(hand, epoch_id="a")
    assert result.action is None and c.fault and c.active_epoch is None
    assert c.activate("b", mono[0] + 0.1).outcome == "fault"
    assert c.step(hand, epoch_id="a").action is None


def test_end_during_solve_fences_action(monkeypatch):
    from dexim.brainco.offline import controller as module

    c, mono, wall = controller()
    mono[0] = 0.11
    original = module.retarget_hand

    def ending(*args):
        c.end("a")
        return original(*args)

    monkeypatch.setattr(module, "retarget_hand", ending)
    assert c.step(synthetic_hand(c.config, wall[0]), epoch_id="a").action is None


def test_bounds_velocity_and_snapshot_copy():
    from dexim.core.robot_interface import JointCommand

    c, mono, wall = controller()
    mono[0] = 0.11
    c.step()
    mono[0] += 0.02
    result = c.step(synthetic_hand(c.config, wall[0], "fist"), epoch_id="a")
    assert max(result.action.positions) <= 0.04 + 1e-9
    state = c.interface.read()
    state.q[:] = 999
    assert max(c.interface.read().q) < 1
    assert np.isnan(state.tau).all()
    for q in ([0], [float("nan")] * 6, [999] * 6):
        with pytest.raises(ValueError):
            c.interface.write(JointCommand(q=np.array(q)))
    with pytest.raises(RuntimeError):
        with c.maintenance():
            pass
    c.shutdown()
    with pytest.raises(RuntimeError):
        c.interface.read()


def test_asset_same_name_alteration_rejected(tmp_path):
    path = tmp_path / "same.urdf"
    path.write_text('<robot name="altered"/>')
    with pytest.raises(ValueError, match="SHA256"):
        describe_model().validate_assets(path)


def test_post_solve_state_age_revokes(monkeypatch):
    from dexim.brainco.offline import controller as module

    mono, wall = [0.0], [100.0]
    cfg = OfflineConfig(state_timeout_s=0.01, solve_budget_s=0.2)
    c = OfflineController(cfg, clock=lambda: mono[0], wall_clock=lambda: wall[0])
    c.prepare()
    c.activate("a", 0.0)
    original = module.retarget_hand

    def delayed(*args):
        q = original(*args)
        mono[0] += 0.02
        wall[0] += 0.02
        return q

    monkeypatch.setattr(module, "retarget_hand", delayed)
    result = c.step(synthetic_hand(cfg, 100.0), epoch_id="a")
    assert result.action is None and "state_stale_after_solve" in c.fault


def test_render_observation_only_and_external_asset_hash(monkeypatch):
    from dexim.brainco.offline import render

    fake_model = Mock()
    fake_model.segments.return_value = np.zeros((6, 2, 3))
    monkeypatch.setattr(render, "NativeModel", Mock(return_value=fake_model))
    scene = Mock()
    renderer = render.ObservedRenderer(describe_model(), "external.urdf", scene)
    c, mono, wall = controller()
    result = c.step()
    renderer.update(result.observation)
    scene.scene.add_line_segments.assert_called_once()
    with pytest.raises(ValueError, match="ObservationRecord"):
        renderer.update(result.action)
    renderer.close()


def test_imports_remain_native_and_sdk_free():
    import subprocess
    import sys

    code = """
import sys
from dexim.brainco.cli import brainco_group
from dexim.brainco.offline import OfflineController
assert not any(n.startswith(('pinocchio', 'viser', 'trimesh', 'bc_stark',
                             'bc_revo3')) for n in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True)
