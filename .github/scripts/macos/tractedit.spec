# -*- mode: python ; coding: utf-8 -*-
"""
TractEdit PyInstaller Spec File for macOS
Build with: pyinstaller .github/scripts/macos/tractedit.spec
"""

import sys
import os
from pathlib import Path
from PIL import Image

block_cipher = None

# Get the project root directory (spec file is in .github/scripts/macos/ subdirectory)
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

# Collect the AOT-compiled numerical extension (.so on macOS)
import glob as _glob
_aot_dir = str(project_root / 'tractedit_pkg' / '_numba_aot')
_aot_binaries = []
for _ext in _glob.glob(os.path.join(_aot_dir, 'tractedit_numba.*')):
    if _ext.endswith('.py') or _ext.endswith('.pyc'):
        continue
    _aot_binaries.append((_ext, 'tractedit_pkg/_numba_aot'))

a = Analysis(
    [str(project_root / 'main.py')],
    pathex=[str(project_root)],
    binaries=_aot_binaries,
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
        'tractedit_pkg.file_io.anatomical',
        'tractedit_pkg.file_io.bundle_gui',
        'tractedit_pkg.file_io.bundle_stats',
        'tractedit_pkg.file_io.bbox_cache',
        'tractedit_pkg.file_io.tractogram',
        'tractedit_pkg.file_io.density_map',
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
        'PyQt6.QtNetwork',
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
        'scipy._cyutility',
        'scipy.sparse',
        'scipy.spatial',
        'nibabel',
        'fury',
        'trx',
        # AOT-compiled numerical extension
        'tractedit_pkg._numba_aot',
        'tractedit_pkg._numba_aot.tractedit_numba',
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
        # Numba is build-time only (AOT extension is pre-compiled)
        'numba',
        'tractedit_pkg._numba_aot.build_aot',
        'tractedit_pkg._numba_aot.kernels',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Exclude unnecessary Qt6 frameworks on macOS
excluded_binaries = [
    'QtWebEngine',
    'QtDesigner',
    'QtQuick',
    'QtQml',
    'QtMultimedia',
    'QtBluetooth',
    'QtNfc',
    'QtSensors',
    'QtSerialPort',
    'QtPositioning',
    'QtLocation',
    'QtTest',
    'QtPdf',
    'QtCharts',
    'QtDataVisualization',
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
    strip=True,  # Strip debug symbols to reduce size
    upx=False,  # UPX disabled - causes issues on macOS ARM64
    console=False,  # No console window for GUI app
    disable_windowed_traceback=False,
    argv_emulation=False,  # Disabled to avoid PyQt6 event loop conflicts
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
    strip=False,  # Don't strip collected binaries - can break code signing on macOS
    upx=False,  # UPX disabled - causes issues on macOS ARM64
    upx_exclude=[],
    name='TractEdit',
)

bundle_icon = Path(workpath) / 'TractEdit.icns'
with Image.open(project_root / 'tractedit_pkg/assets/logo.ico') as source_icon:
    source_icon.save(bundle_icon)

app = BUNDLE(
    coll,
    name='TractEdit.app',
    icon=str(bundle_icon),
    bundle_identifier='com.tractedit.app',
    info_plist={
        'CFBundleName': 'TractEdit',
        'CFBundleDisplayName': 'TractEdit',
        'CFBundleVersion': '3.6.0',
        'CFBundleShortVersionString': '3.6.0',
        'NSHighResolutionCapable': True,
        'NSRequiresAquaSystemAppearance': False,
        'UTExportedTypeDeclarations': [{
            'UTTypeIdentifier': 'com.tractedit.session',
            'UTTypeDescription': 'TractEdit Session',
            'UTTypeConformsTo': ['com.pkware.zip-archive'],
            'UTTypeTagSpecification': {'public.filename-extension': ['tractedit-session']},
            'UTTypeIconFile': 'TractEdit.icns',
        }],
        'CFBundleDocumentTypes': [
            {
                'CFBundleTypeName': 'Wavefront OBJ Mesh',
                'CFBundleTypeExtensions': ['obj'],
                'CFBundleTypeRole': 'Viewer',
                'LSHandlerRank': 'Alternate',
            },
            {
                'CFBundleTypeName': 'TractEdit Session',
                'CFBundleTypeExtensions': ['tractedit-session'],
                'CFBundleTypeIconFile': 'TractEdit.icns',
                'CFBundleTypeRole': 'Editor',
                'LSItemContentTypes': ['com.tractedit.session'],
                'LSHandlerRank': 'Owner',
            },
            {
                'CFBundleTypeName': 'NIfTI Image',
                'CFBundleTypeExtensions': ['nii'],
                'CFBundleTypeIconFile': 'TractEdit.icns',
                'CFBundleTypeRole': 'Viewer',
                'LSHandlerRank': 'Alternate',
            },
            {
                'CFBundleTypeName': 'Compressed NIfTI Image',
                'CFBundleTypeExtensions': ['nii.gz'],
                'CFBundleTypeRole': 'Viewer',
                'LSHandlerRank': 'Alternate',
            },
            {
                'CFBundleTypeName': 'TRK File',
                'CFBundleTypeExtensions': ['trk'],
                'CFBundleTypeIconFile': 'TractEdit.icns',
                'CFBundleTypeRole': 'Editor',
            },
            {
                'CFBundleTypeName': 'TCK File',
                'CFBundleTypeExtensions': ['tck'],
                'CFBundleTypeIconFile': 'TractEdit.icns',
                'CFBundleTypeRole': 'Editor',
            },
            {
                'CFBundleTypeName': 'TRX File',
                'CFBundleTypeExtensions': ['trx'],
                'CFBundleTypeIconFile': 'TractEdit.icns',
                'CFBundleTypeRole': 'Editor',
            },
            {
                'CFBundleTypeName': 'VTK File',
                'CFBundleTypeExtensions': ['vtk'],
                'CFBundleTypeIconFile': 'TractEdit.icns',
                'CFBundleTypeRole': 'Editor',
            },
            {
                'CFBundleTypeName': 'VTP File',
                'CFBundleTypeExtensions': ['vtp'],
                'CFBundleTypeIconFile': 'TractEdit.icns',
                'CFBundleTypeRole': 'Editor',
            },
        ],
    },
)
