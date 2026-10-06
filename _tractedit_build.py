"""Poetry build hook for the TractEdit AOT extension."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
AOT_SCRIPT = PROJECT_ROOT / "tractedit_pkg" / "_numba_aot" / "build_aot.py"


def compile_aot() -> None:
    """Compile the required platform-specific numerical extension."""
    if not AOT_SCRIPT.is_file():
        raise FileNotFoundError(f"AOT build script not found: {AOT_SCRIPT}")

    result = subprocess.run(
        [sys.executable, str(AOT_SCRIPT)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return

    output = "\n".join(
        part.strip() for part in (result.stdout, result.stderr) if part.strip()
    )
    raise RuntimeError(
        f"TractEdit AOT compilation failed with exit code "
        f"{result.returncode}.\n{output}"
    )


if __name__ == "__main__":
    compile_aot()
