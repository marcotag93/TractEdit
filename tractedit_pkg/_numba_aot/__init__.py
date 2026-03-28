# -*- coding: utf-8 -*-

"""
AOT-compiled numerical kernels for TractEdit.

This module re-exports every function from the compiled extension
(tractedit_numba.pyd / .so). If the extension is missing, the
application cannot start — run the build step first.
"""

import sys as _sys

# When invoked as ``python tractedit_pkg/_numba_aot/build_aot.py``,
# the compiled module does not yet exist.  Skip the import so the
# build script can run without a chicken-and-egg failure.
_is_building = any(
    arg.endswith("build_aot") or arg.endswith("build_aot.py") for arg in _sys.argv
)

if not _is_building:
    try:
        from .tractedit_numba import *  # noqa: F401,F403
    except ImportError as e:
        raise ImportError(
            "\n"
            "========================================================\n"
            "  TractEdit AOT module not found.\n"
            "  The compiled numerical extension is required to run.\n"
            "\n"
            "  Build it with:\n"
            "    poetry install --with build\n"
            "    python tractedit_pkg/_numba_aot/build_aot.py\n"
            "\n"
            "  Then restart the application.\n"
            "========================================================"
        ) from e
