# -*- coding: utf-8 -*-

"""
AOT compilation script for TractEdit Numba functions.

Compiles all performance-critical numerical kernels into a platform-specific
shared library (tractedit_numba.pyd on Windows, .so on Unix) using Numba's
Ahead-of-Time compilation via ``numba.pycc.CC``.

Usage:
    python tractedit_pkg/_numba_aot/build_aot.py

Output:
    tractedit_pkg/_numba_aot/tractedit_numba.<ext>

The compiled module is imported at runtime by ``__init__.py`` without
requiring Numba to be installed.

Notes:
    On Windows, if MSVC is not in PATH, the script will attempt to locate
    Visual Studio Build Tools and activate the environment automatically.
    You can also set ``DISTUTILS_USE_SDK=1`` and ``MSSdk=1`` manually after
    running ``vcvars64.bat``.
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np  # noqa: F401 — required by Numba type resolution
from numba.core.compiler import Flags
from numba.pycc import CC
from numba.pycc import compiler as pycc_compiler

# ---------------------------------------------------------------------------
# Output directory: place the compiled extension next to this script.
# ---------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if __package__:
    from .kernels import (
        register_mesh_validation_kernels,
        register_obj_kernels,
        register_streamline_kernels,
    )
else:
    if _SCRIPT_DIR not in sys.path:
        sys.path.insert(0, _SCRIPT_DIR)
    from kernels import (
        register_mesh_validation_kernels,
        register_obj_kernels,
        register_streamline_kernels,
    )

# ---------------------------------------------------------------------------
# Windows MSVC auto-detection
# ---------------------------------------------------------------------------


def _ensure_msvc_env() -> None:
    """Activate MSVC environment on Windows if not already available.

    Sets ``DISTUTILS_USE_SDK`` and ``MSSdk`` so that ``distutils``
    picks up the pre-configured SDK environment rather than trying
    (and failing) to auto-detect Visual Studio installations.
    """
    if sys.platform != "win32":
        return

    # Already configured?
    if os.environ.get("DISTUTILS_USE_SDK") == "1":
        return

    # Try to find vcvarsall.bat
    patterns = [
        r"C:\Program Files (x86)\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvarsall.bat",
        r"C:\Program Files\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvarsall.bat",
    ]
    vcvarsall = None
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            vcvarsall = matches[-1]  # Latest version
            break

    if vcvarsall is None:
        print(
            "WARNING: Could not find vcvarsall.bat. "
            "MSVC may not be detected by distutils.",
            file=sys.stderr,
        )
        return

    print(f"Activating MSVC environment via: {vcvarsall}")

    # Run vcvarsall.bat and capture the resulting environment.
    # We use cmd.exe explicitly with a fixed argument list to avoid
    # shell injection via the vcvarsall path.
    try:
        result = subprocess.run(
            ["cmd.exe", "/C", vcvarsall, "amd64", ">nul", "2>&1", "&&", "set"],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"MSVC environment activation timed out after 120 seconds: {vcvarsall}"
        ) from error
    if result.returncode != 0:
        print(
            f"WARNING: vcvarsall.bat failed (exit {result.returncode})",
            file=sys.stderr,
        )
        return

    # Only import the environment variables required by distutils/MSVC.
    _MSVC_ENV_WHITELIST = {
        "INCLUDE",
        "LIB",
        "LIBPATH",
        "PATH",
        "VSINSTALLDIR",
        "VCINSTALLDIR",
        "WINDOWSSDKDIR",
        "WINDOWSSDKVERSION",
        "UCRTVERSION",
        "VCTOOLSINSTALLDIR",
        "VCTOOLSREDISTDIR",
        "VSCMD_ARG_HOST_ARCH",
        "VSCMD_ARG_TGT_ARCH",
    }
    for line in result.stdout.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            if key in _MSVC_ENV_WHITELIST:
                os.environ[key] = value

    os.environ["DISTUTILS_USE_SDK"] = "1"
    os.environ["MSSdk"] = "1"
    print("MSVC environment activated successfully.")


# ---------------------------------------------------------------------------
# Compiler setup
# ---------------------------------------------------------------------------


def _create_compiler() -> CC:
    """Create and configure the Numba AOT compiler."""
    compiler = CC("tractedit_numba")
    compiler.verbose = True

    # Use 'host' for local dev builds (fastest on the build machine).
    # CI/release builds should override to 'generic' for portability.
    compiler.target_cpu = os.environ.get("TRACTEDIT_AOT_TARGET_CPU", "host")

    # Place the compiled extension next to this script.
    compiler.output_dir = _SCRIPT_DIR

    return compiler


@contextmanager
def _gil_releasing_compiler_flags() -> Iterator[None]:
    """Build CPython wrappers that release the GIL during native calls."""
    original_factory = pycc_compiler.Flags

    def create_flags() -> Flags:
        flags = Flags()
        flags.release_gil = True
        return flags

    pycc_compiler.Flags = create_flags
    try:
        yield
    finally:
        pycc_compiler.Flags = original_factory


# ---------------------------------------------------------------------------
# AOT function registration
# ---------------------------------------------------------------------------


def _register_functions(compiler: CC) -> None:
    """Register all AOT-exported functions on the given CC instance."""
    register_streamline_kernels(compiler)
    register_mesh_validation_kernels(compiler)
    register_obj_kernels(compiler)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Build the AOT extension module."""
    _ensure_msvc_env()

    compiler = _create_compiler()
    _register_functions(compiler)

    print(f"Compiling tractedit_numba -> {_SCRIPT_DIR}")
    with _gil_releasing_compiler_flags():
        compiler.compile()
    print("AOT compilation completed successfully.")


if __name__ == "__main__":
    main()
