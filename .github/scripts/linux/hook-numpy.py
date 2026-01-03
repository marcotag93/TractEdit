# -*- coding: utf-8 -*-
"""
PyInstaller runtime hook to fix numpy "source directory" detection issue.
Runs before the application to prevent false positives in bundled apps.
"""
import sys
import os

# Set environment variable early
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '1'


def patch_numpy_source_detection():
    """Remove or rename files that trigger numpy's source directory detection."""
    if not getattr(sys, 'frozen', False):
        return

    base_path = sys._MEIPASS
    
    # Check both MEIPASS root and _internal subdirectory
    search_paths = [base_path]
    internal_path = os.path.join(base_path, '_internal')
    if os.path.exists(internal_path):
        search_paths.append(internal_path)
    
    for search_base in search_paths:
        problematic_files = [
            os.path.join(search_base, 'setup.py'),
            os.path.join(search_base, 'pyproject.toml'),
            os.path.join(search_base, 'numpy', 'setup.py'),
            os.path.join(search_base, 'numpy', 'pyproject.toml'),
        ]

        for filepath in problematic_files:
            if os.path.exists(filepath):
                try:
                    os.rename(filepath, filepath + '.disabled')
                except (OSError, PermissionError):
                    pass 


def cleanup_source_paths():
    """Remove source-tree-like paths from sys.path."""
    if not getattr(sys, 'frozen', False):
        return

    base_path = sys._MEIPASS
    internal_path = os.path.join(base_path, '_internal')

    # Ensure _internal is first in path
    if os.path.exists(internal_path) and internal_path not in sys.path:
        sys.path.insert(0, internal_path)

    # Filter out paths
    cleaned_path = []
    for p in sys.path:
        if os.path.isdir(p):
            setup_py = os.path.join(p, 'setup.py')
            pyproject = os.path.join(p, 'pyproject.toml')
            numpy_dir = os.path.join(p, 'numpy')
            
            # Skip paths 
            if os.path.exists(numpy_dir) and (os.path.exists(setup_py) or os.path.exists(pyproject)):
                continue
        cleaned_path.append(p)
    
    sys.path[:] = cleaned_path


# Execute patches
patch_numpy_source_detection()
cleanup_source_paths()
