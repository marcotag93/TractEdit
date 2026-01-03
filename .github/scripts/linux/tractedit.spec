# -*- mode: python ; coding: utf-8 -*-
"""
TractEdit PyInstaller Spec File for Linux
Build with: pyinstaller .github/scripts/linux/tractedit.spec
"""

import sys
import os
from pathlib import Path

block_cipher = None

# Get the project root directory (spec file is in .github/scripts/linux/ subdirectory)
project_root = Path(SPECPATH).parent.parent.parent

# Find fury package location for stub files
import fury
fury_path = os.path.dirname(fury.__file__)

# Find numpy package location for __config__ module
import numpy
numpy_path = os.path.dirname(numpy.__file__)

# Collect numpy __config__.py if it exists (this is a generated file)
numpy_config_files = []
numpy_config_path = os.path.join(numpy_path, '__config__.py')
if os.path.exists(numpy_config_path):
    numpy_config_files.append((numpy_config_path, 'numpy'))

# Collect numpy.libs directory if it exists (contains required DLLs/SOs)
numpy_libs_path = os.path.join(os.path.dirname(numpy_path), 'numpy.libs')
if os.path.exists(numpy_libs_path):
    numpy_config_files.append((numpy_libs_path, 'numpy.libs'))

# Collect all fury .pyi stub files
fury_stubs = []
for root, dirs, files in os.walk(fury_path):
    for f in files:
        if f.endswith('.pyi'):
            src = os.path.join(root, f)
            # Get relative path from fury package
            rel_path = os.path.relpath(root, fury_path)
            if rel_path == '.':
                dest = 'fury'
            else:
                dest = os.path.join('fury', rel_path)
            fury_stubs.append((src, dest))

a = Analysis(
    [str(project_root / 'main.py')],
    pathex=[str(project_root)],
    binaries=[],
    datas=[
        # Include all assets
        (str(project_root / 'tractedit_pkg' / 'assets'), 'tractedit_pkg/assets'),
        # Include fury data directory
        (os.path.join(fury_path, 'data'), 'fury/data'),
    ] + fury_stubs + numpy_config_files,  # Add all fury stub files and numpy config
    hiddenimports=[
        'tractedit_pkg',
        'tractedit_pkg.assets',
        'tractedit_pkg.main_window',
        'tractedit_pkg.file_io',
        'tractedit_pkg.utils',
        'tractedit_pkg.odf_utils',
        'tractedit_pkg.logic',
        'tractedit_pkg.logic.connectivity',
        'tractedit_pkg.logic.roi_manager',
        'tractedit_pkg.logic.scalar_manager',
        'tractedit_pkg.logic.state_manager',
        'tractedit_pkg.ui',
        'tractedit_pkg.visualization',
        # PyQt6 modules
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
        # VTK - comprehensive imports
        'vtkmodules',
        'vtkmodules.all',
        'vtkmodules.util',
        'vtkmodules.util.numpy_support',
        'vtkmodules.vtkCommonCore',
        'vtkmodules.vtkCommonDataModel',
        'vtkmodules.vtkCommonExecutionModel',
        'vtkmodules.vtkCommonMath',
        'vtkmodules.vtkCommonTransforms',
        'vtkmodules.vtkFiltersCore',
        'vtkmodules.vtkFiltersGeneral',
        'vtkmodules.vtkFiltersModeling',
        'vtkmodules.vtkFiltersSources',
        'vtkmodules.vtkIOCore',
        'vtkmodules.vtkIOImage',
        'vtkmodules.vtkIOLegacy',
        'vtkmodules.vtkIOXML',
        'vtkmodules.vtkInteractionStyle',
        'vtkmodules.vtkInteractionWidgets',
        'vtkmodules.vtkRenderingAnnotation',
        'vtkmodules.vtkRenderingCore',
        'vtkmodules.vtkRenderingFreeType',
        'vtkmodules.vtkRenderingOpenGL2',
        'vtkmodules.vtkRenderingUI',
        'vtkmodules.vtkRenderingVolume',
        'vtkmodules.vtkRenderingVolumeOpenGL2',
        # Scientific stack
        'numpy',
        'numpy.__config__',
        'numpy._core',
        'numpy._distributor_init',
        'scipy',
        'nibabel',
        'numba',
        'fury',
        'trx',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(Path(SPECPATH) / 'hook-numpy.py')],
    excludes=[
        # GUI frameworks we don't use
        'tkinter',
        'PyQt5',
        'PySide2',
        'PySide6',
        # Development/testing tools
        'matplotlib',
        'IPython',
        'jupyter',
        'pytest',
        'sphinx',
        'docutils',
        # Unused heavy packages
        'pandas',
        'cv2',
        'torch',
        'tensorflow',
        'keras',
        'sklearn',
        'skimage',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Exclude problematic and unnecessary libraries
excluded_binaries = [
    # System libraries (use system versions)
    'libtiff.so',
    'libjpeg.so',
    'libpng.so',
    'libpng16.so',
    'libwebp.so',
    'libz.so',
    # Large unused libraries
    'libQt6WebEngine',
    'libQt6Designer',
    'libQt6Quick',
    'libQt6Qml',
    'libQt6Multimedia',
    'libQt6Bluetooth',
    'libQt6Nfc',
    'libQt6Sensors',
    'libQt6SerialPort',
    'libQt6Positioning',
    'libQt6Location',
    'libQt6Test',
    'libQt6Pdf',
    'libQt6Charts',
    'libQt6DataVisualization',
]

def should_exclude(name):
    """Check if a binary should be excluded."""
    name_lower = name.lower()
    for excl in excluded_binaries:
        if excl.lower() in name_lower:
            return True
    return False

# Filter out unnecessary binaries
a.binaries = [b for b in a.binaries if not should_exclude(b[0])]

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='tractedit',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,  # Disabled: stripping can cause ELF alignment issues with scipy/OpenBLAS
    upx=True,
    console=False,  # No console window for GUI app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(project_root / 'tractedit_pkg' / 'assets' / 'tractedit.png'),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,  # Disabled: stripping can cause ELF alignment issues with scipy/OpenBLAS
    upx=True,
    upx_exclude=[],
    name='tractedit',
)
