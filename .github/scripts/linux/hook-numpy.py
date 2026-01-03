# -*- coding: utf-8 -*-
"""
PyInstaller runtime hook to fix numpy "source directory" detection issue.
Runs before the application to prevent false positives in bundled apps.

The core issue: numpy 2.x checks for numpy.__config__ module during import.
If it can't find it, numpy assumes you're importing from source directory
and raises ImportError. This hook creates a dummy __config__ module if missing.
"""
import sys
import os
import types

# Set environment variable early
os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "1"


def create_dummy_numpy_config():
    """
    Create a dummy numpy.__config__ module if it doesn't exist.

    If this import fails with ModuleNotFoundError for "numpy.__config__",
    numpy raises the "source directory" error. We pre-create this module
    to prevent that check from failing.
    """
    if not getattr(sys, "frozen", False):
        return

    # Check if numpy.__config__ already exists in sys.modules
    if "numpy.__config__" in sys.modules:
        _log("numpy.__config__ already in sys.modules")
        return

    # Create a dummy __config__ module
    config_module = types.ModuleType("numpy.__config__")

    # Add the required show_config function (numpy imports this)
    def show_config():
        """Dummy show_config for PyInstaller bundle."""
        print("NumPy configuration (PyInstaller bundle)")
        print("  Build environment: PyInstaller frozen app")
        print("  No detailed config available in bundled app")

    config_module.show_config = show_config

    # Add other attributes numpy might check
    config_module.__file__ = "<frozen numpy.__config__>"
    config_module.__doc__ = "Dummy numpy config for PyInstaller bundle"

    # Pre-register in sys.modules BEFORE numpy tries to import it
    sys.modules["numpy.__config__"] = config_module


def patch_numpy_source_detection():
    """Remove or rename files that trigger numpy's source directory detection."""
    if not getattr(sys, "frozen", False):
        return

    base_path = sys._MEIPASS

    # Check both MEIPASS root and _internal subdirectory
    search_paths = [base_path]
    internal_path = os.path.join(base_path, "_internal")
    if os.path.exists(internal_path):
        search_paths.append(internal_path)

    for search_base in search_paths:
        problematic_files = [
            os.path.join(search_base, "setup.py"),
            os.path.join(search_base, "pyproject.toml"),
            os.path.join(search_base, "numpy", "setup.py"),
            os.path.join(search_base, "numpy", "pyproject.toml"),
        ]

        for filepath in problematic_files:
            if os.path.exists(filepath):
                try:
                    os.rename(filepath, filepath + ".disabled")
                except (OSError, PermissionError):
                    pass  # Silently ignore rename failures


def cleanup_source_paths():
    """Remove source-tree-like paths from sys.path."""
    if not getattr(sys, "frozen", False):
        return

    base_path = sys._MEIPASS
    internal_path = os.path.join(base_path, "_internal")

    # Ensure _internal is first in path
    if os.path.exists(internal_path) and internal_path not in sys.path:
        sys.path.insert(0, internal_path)

    # Filter out paths that look like source directories
    cleaned_path = []
    for p in sys.path:
        if os.path.isdir(p):
            setup_py = os.path.join(p, "setup.py")
            pyproject = os.path.join(p, "pyproject.toml")
            numpy_dir = os.path.join(p, "numpy")

            # Skip paths that look like numpy source tree
            if os.path.exists(numpy_dir) and (
                os.path.exists(setup_py) or os.path.exists(pyproject)
            ):
                continue
        cleaned_path.append(p)

    sys.path[:] = cleaned_path


# Execute patches in order of priority
# 1. Create dummy __config__ module FIRST
create_dummy_numpy_config()

# 2. Rename problematic files
patch_numpy_source_detection()

# 3. Clean up sys.path
cleanup_source_paths()
