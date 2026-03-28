# -*- coding: utf-8 -*-

"""
Custom PEP 517 build backend wrapping poetry-core.

Ensures the AOT numerical extension is compiled before the wheel is
assembled, so ``pip install .`` always produces a working package.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Re-export the full PEP 517 interface from poetry-core so that pip,
# build, and other frontends see every hook they expect.
from poetry.core.masonry.api import *  # noqa: F401,F403
from poetry.core.masonry.api import build_sdist as _poetry_build_sdist
from poetry.core.masonry.api import build_wheel as _poetry_build_wheel

_AOT_SCRIPT = (
    Path(__file__).resolve().parent
    / "tractedit_pkg"
    / "_numba_aot"
    / "build_aot.py"
)


def _compile_aot() -> None:
    """Compile the AOT numerical extension.

    Raises on failure — never silently skips.
    """
    if not _AOT_SCRIPT.exists():
        raise FileNotFoundError(
            f"AOT build script not found at {_AOT_SCRIPT}. "
            "Ensure the source tree is complete."
        )

    print(f"[TractEdit] Compiling AOT extension via {_AOT_SCRIPT} ...")
    result = subprocess.run(
        [sys.executable, str(_AOT_SCRIPT)],
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"[TractEdit] AOT compilation failed "
            f"(exit code {result.returncode}).\n"
            f"Compiler output:\n{result.stderr}\n"
            "Hint: ensure a C compiler is available "
            "(MSVC on Windows, gcc/clang on Unix)."
        )
    print("[TractEdit] AOT extension compiled successfully.")


def build_wheel(
    wheel_directory,
    config_settings=None,
    metadata_directory=None,
):
    """Compile AOT extension, then build the wheel."""
    _compile_aot()
    return _poetry_build_wheel(
        wheel_directory,
        config_settings=config_settings,
        metadata_directory=metadata_directory,
    )


def build_sdist(sdist_directory, config_settings=None):
    """Build sdist — AOT compilation is deferred to wheel build."""
    return _poetry_build_sdist(
        sdist_directory, config_settings=config_settings
    )
