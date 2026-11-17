"""Bounded SDK-free model/config checks and offline demonstration."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, replace

import rich_click as click

from dexim.brainco.offline import (
    OfflineConfig,
    OfflineController,
    describe_model,
    load_replay,
    synthetic_hand,
)


@click.command("offline")
@click.option("--model", type=click.Choice(["revo2", "revo3"]), required=True)
@click.option("--side", type=click.Choice(["left", "right"]), required=True)
@click.option("--steps", type=click.IntRange(1, 1000), default=10)
@click.option("--replay", type=click.Path(exists=True, dir_okay=False))
@click.option("--check", is_flag=True, help="Print validated model/config and exit.")
def offline(model: str, side: str, steps: int, replay: str | None, check: bool) -> None:
    """Run a finite mock demo; no devices, SDK, scans or calibration."""
    config = OfflineConfig(model=model, side=side)
    try:
        records = load_replay(replay, config) if replay else None
        if check:
            click.echo(
                json.dumps(
                    {
                        "config": asdict(config),
                        "model": asdict(describe_model(model, side)),
                    }
                )
            )
            return
        # Deterministic control time; input capture time remains real Unix time.
        tick = [0.0]
        controller = OfflineController(config, clock=lambda: tick[0])
        controller.prepare()
        controller.activate("offline-1", 0.0)
        for i in range(min(steps, len(records)) if records else steps):
            tick[0] = i * 0.02
            controller.watchdog("offline-1", i)
            hand = (
                replace(records[i], timestamp=time.time())
                if records
                else synthetic_hand(config, time.time(), "fist")
            )
            result = controller.step(hand, epoch_id="offline-1")
            payload = asdict(result)
            if records:
                payload["replay"] = {"original_timestamp": records[i].timestamp}
            click.echo(json.dumps(payload))
            if result.report.fault:
                raise click.ClickException(result.report.fault)
        controller.end("offline-1")
        controller.shutdown()
    except (ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc
