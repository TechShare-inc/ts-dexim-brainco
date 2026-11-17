"""Bounded CLI subprocess checks without device or native dependency loading."""

import json
import subprocess
import sys

import pytest


@pytest.mark.parametrize("model,side,n", [("revo2", "left", 6), ("revo3", "right", 21)])
def test_bounded_cli_streams(model, side, n):
    script = "from dexim.brainco.cli import brainco_group; brainco_group()"
    run = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            "offline",
            "--model",
            model,
            "--side",
            side,
            "--steps",
            "3",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    rows = [json.loads(line) for line in run.stdout.splitlines()]
    assert len(rows) == 3
    assert all(len(row["action"]["positions"]) == n for row in rows)
    assert all(row["observation"]["feedback"] == "mock" for row in rows)
    assert all(row["action"]["epoch_id"] == "offline-1" for row in rows)


def test_legacy_config_rejects_revo3_before_activation():
    from dexim.brainco.node.config import BrainCoConfig

    with pytest.raises(ValueError, match="Legacy"):
        BrainCoConfig(model="revo3")
