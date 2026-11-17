"""Test the built wheel in an isolated venv using the locked dependencies."""

from __future__ import annotations

import os
import subprocess
import tempfile
import venv
from pathlib import Path


def main() -> None:
    """Install exactly one wheel and run the offline suite against its bytes."""
    root = Path(__file__).resolve().parents[1]
    wheels = list((root / "dist").glob("dexim_brainco-*.whl"))
    if len(wheels) != 1:
        raise SystemExit("Build exactly one BrainCo wheel in dist first.")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    with tempfile.TemporaryDirectory(prefix="brainco-wheel-") as directory:
        target = Path(directory)
        venv.create(target, with_pip=True, system_site_packages=True)
        python = target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--ignore-installed",
                str(wheels[0]),
            ],
            check=True,
            env=env,
        )
        subprocess.run(
            [
                str(python),
                "-c",
                "import pathlib, sys; import dexim.brainco.offline as module; "
                "assert pathlib.Path(module.__file__).is_relative_to(sys.prefix), "
                "module.__file__; print('Installed wheel:', module.__file__)",
            ],
            check=True,
            env=env,
            cwd=directory,
        )
        subprocess.run(
            [
                str(python),
                "-m",
                "pytest",
                str(root / "tests"),
                "-m",
                "not hardware and not requires_model",
            ],
            check=True,
            env=env,
            cwd=directory,
        )


if __name__ == "__main__":
    main()
