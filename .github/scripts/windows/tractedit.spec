# -*- mode: python ; coding: utf-8 -*-
"""
TractEdit PyInstaller Spec File for Windows
Build with: pyinstaller .github/scripts/windows/tractedit.spec --noconfirm
"""

import sys
import os
from pathlib import Path

block_cipher = None

# Get the project root directory (spec file is in .github/scripts/windows/ subdirectory)
project_root = Path(SPECPATH).parent.parent.parent

# Find fury package location for stub files
import fury
fury_path = os.path.dirname(fury.__file__)

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
    ] + fury_stubs,  # Add all fury stub files
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
        'scipy',
        'scipy.special._cdflib',
        'nibabel',
        'numba',
        'fury',
        'trx',
        # Lazy loader (required by some dependencies)
        'lazy_loader',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
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

# Exclude problematic and unnecessary libraries on Windows
excluded_binaries = [
    # Large unused Qt6 modules (DLLs on Windows)
    'Qt6WebEngine',
    'Qt6Designer',
    'Qt6Quick',
    'Qt6Qml',
    'Qt6Multimedia',
    'Qt6Bluetooth',
    'Qt6Nfc',
    'Qt6Sensors',
    'Qt6SerialPort',
    'Qt6Positioning',
    'Qt6Location',
    'Qt6Test',
    'Qt6Pdf',
    'Qt6Charts',
    'Qt6DataVisualization',
    'Qt6RemoteObjects',
    'Qt6Scxml',
    'Qt6StateMachine',
    'Qt6VirtualKeyboard',
    'Qt63DCore',
    'Qt63DRender',
    'Qt63DInput',
    'Qt63DLogic',
    'Qt63DAnimation',
    'Qt63DExtras',
    # WebEngine is particularly large (~100-200MB)
    'QtWebEngine',
    'webengine',
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
    name='TractEdit',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,  # Windows doesn't support strip like Unix
    upx=True,  # Enable UPX compression (if UPX is available)
    console=False,  # No console window for GUI app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(project_root / 'tractedit_pkg' / 'assets' / 'logo.ico'),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,  # Windows doesn't support strip
    upx=True,  # Enable UPX compression on collected binaries
    upx_exclude=[
        # Exclude files that don't compress well or cause issues with UPX
        'vcruntime140.dll',
        'vcruntime140_1.dll',
        'msvcp140.dll',
        'python*.dll',
        'api-ms-*.dll',
        'ucrtbase.dll',
    ],
    name='TractEdit',
)
