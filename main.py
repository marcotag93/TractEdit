# -*- coding: utf-8 -*-

"""
Tractedit GUI - Main Application Runner
"""

# ============================================================================
# Imports
# ============================================================================

import os
import sys
import multiprocessing

# PyInstaller freeze support
if __name__ == "__main__":
    multiprocessing.freeze_support()

if sys.platform == "darwin":  # macOS
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import logging
import importlib.resources
import ctypes
from PyQt6.QtWidgets import QApplication, QSplashScreen
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor, QFont
import argparse
from PyQt6.QtCore import Qt, QRect, QTimer

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ============================================================================
# Splash Screen
# ============================================================================


class LoadingSplash(QSplashScreen):
    """
    Custom Splash Screen with a Progress Bar drawn at the bottom.
    """

    def __init__(self, pixmap, flags=Qt.WindowType.WindowStaysOnTopHint):
        super().__init__(pixmap, flags)
        self.progress = 0
        self.message = "Initializing..."

        # UI Settings
        self.progress_height = 20
        self.bar_color = QColor(135, 206, 250)  # progress bar color
        self.text_color = QColor(135, 206, 250)  # progress bar text
        self.setCursor(Qt.CursorShape.WaitCursor)

    def set_progress(self, value, message=None):
        self.progress = value
        if message:
            self.message = message
        self.repaint()  # Force a redraw
        QApplication.processEvents()

    def drawContents(self, painter: QPainter):

        # Draw the Pixmap (Logo)
        super().drawContents(painter)

        # Setup Geometry
        rect = self.rect()
        text_space_height = 30

        bar_y_pos = rect.height() - self.progress_height - text_space_height

        bar_rect = QRect(
            0,
            bar_y_pos,
            int(rect.width() * (self.progress / 100)),
            self.progress_height,
        )

        text_rect = QRect(
            0, rect.height() - text_space_height, rect.width(), text_space_height
        )

        # Draw Progress Bar
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self.bar_color)
        painter.drawRect(bar_rect)

        # Draw Loading Text
        painter.setPen(self.text_color)
        painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, self.message)


# ============================================================================
# Headless Operations
# ============================================================================


def _run_headless_conversion(input_path: str, output_path: str) -> None:
    """
    Performs headless format conversion without initializing the GUI.

    Args:
        input_path: Path to the input bundle file.
        output_path: Path to the output file (format determined by extension).

    Raises:
        SystemExit: If input/output paths are missing or conversion fails.
    """
    import os
    import numpy as np

    # Validate arguments
    if not input_path:
        logger.error("Error: Input bundle file is required for --convert-to.")
        print("Error: Input bundle file is required for --convert-to.", file=sys.stderr)
        print("Usage: tractedit input.trk --convert-to output.trx", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(input_path):
        logger.error(f"Error: Input file not found: {input_path}")
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Get file extensions
    input_ext = os.path.splitext(input_path)[1].lower()
    output_ext = os.path.splitext(output_path)[1].lower()

    # Handle .nii.gz special case
    if output_path.lower().endswith(".nii.gz"):
        output_ext = ".nii.gz"

    # Validate extensions
    valid_input_exts = {".trk", ".tck", ".trx", ".vtk", ".vtp"}
    valid_output_exts = {".trk", ".tck", ".trx", ".vtk", ".vtp"}

    if input_ext not in valid_input_exts:
        logger.error(f"Error: Unsupported input format: {input_ext}")
        print(f"Error: Unsupported input format: {input_ext}", file=sys.stderr)
        print(f"Supported formats: {', '.join(valid_input_exts)}", file=sys.stderr)
        sys.exit(1)

    if output_ext not in valid_output_exts:
        logger.error(f"Error: Unsupported output format: {output_ext}")
        print(f"Error: Unsupported output format: {output_ext}", file=sys.stderr)
        print(f"Supported formats: {', '.join(valid_output_exts)}", file=sys.stderr)
        sys.exit(1)

    print(f"Converting: {input_path} -> {output_path}")
    logger.info(f"Headless conversion: {input_path} -> {output_path}")

    try:
        import nibabel as nib
        from nibabel.streamlines import Field
        import trx.trx_file_memmap as tbx

        # Suppress verbose INFO logs from trx library
        root_logger = logging.getLogger()
        original_level = root_logger.level
        root_logger.setLevel(logging.WARNING)

        # Load input file
        streamlines = None
        header = None
        affine = None

        if input_ext == ".trk":
            trk = nib.streamlines.load(input_path)
            streamlines = list(trk.streamlines)
            header = dict(trk.header)
            affine = trk.affine

        elif input_ext == ".tck":
            tck = nib.streamlines.load(input_path)
            streamlines = list(tck.streamlines)
            # TCK doesn't have affine, use identity
            affine = np.eye(4)

        elif input_ext == ".trx":
            trx_file = tbx.load(input_path)
            streamlines = list(trx_file.streamlines)
            affine = trx_file.header.get("VOXEL_TO_RASMM", np.eye(4))
            if hasattr(trx_file, "header") and "DIMENSIONS" in trx_file.header:
                header = {"dimensions": trx_file.header["DIMENSIONS"]}

        elif input_ext in (".vtk", ".vtp"):
            import vtk
            from vtk.util import numpy_support

            if input_ext == ".vtk":
                reader = vtk.vtkPolyDataReader()
            else:
                reader = vtk.vtkXMLPolyDataReader()

            reader.SetFileName(input_path)
            reader.Update()
            polydata = reader.GetOutput()

            lines = polydata.GetLines()
            points = numpy_support.vtk_to_numpy(polydata.GetPoints().GetData())

            streamlines = []
            lines.InitTraversal()
            id_list = vtk.vtkIdList()
            while lines.GetNextCell(id_list):
                n_pts = id_list.GetNumberOfIds()
                sl = np.zeros((n_pts, 3))
                for i in range(n_pts):
                    sl[i] = points[id_list.GetId(i)]
                streamlines.append(sl)
            affine = np.eye(4)

        if not streamlines:
            raise ValueError("No streamlines loaded from input file.")

        print(f"Loaded {len(streamlines)} streamlines.")

        # Save output file
        if output_ext == ".trk":
            # Create TRK tractogram
            tractogram = nib.streamlines.Tractogram(
                streamlines=streamlines, affine_to_rasmm=np.eye(4)
            )
            # Use header dimensions if available
            if header and "dimensions" in header:
                dimensions = header["dimensions"]
            else:
                # Compute from streamlines
                all_pts = np.concatenate(streamlines)
                dimensions = np.ceil(np.max(all_pts, axis=0) + 1).astype(int)

            trk_header = {
                Field.VOXEL_TO_RASMM: affine,
                Field.DIMENSIONS: dimensions,
                Field.VOXEL_SIZES: np.abs(np.diag(affine)[:3]),
            }
            trk_file = nib.streamlines.TrkFile(tractogram, header=trk_header)
            nib.streamlines.save(trk_file, output_path)

        elif output_ext == ".tck":
            tractogram = nib.streamlines.Tractogram(
                streamlines=streamlines, affine_to_rasmm=np.eye(4)
            )
            tck_file = nib.streamlines.TckFile(tractogram)
            nib.streamlines.save(tck_file, output_path)

        elif output_ext == ".trx":
            # TRX requires a reference NIfTI image that defines the space
            # Get reference dimensions
            if header and "dimensions" in header:
                dimensions = tuple(np.array(header["dimensions"]).astype(int))
            else:
                all_pts = np.concatenate(streamlines)
                dimensions = tuple(np.ceil(np.max(all_pts, axis=0) + 1).astype(int))

            voxel_sizes = np.abs(np.diag(affine)[:3])
            if np.all(voxel_sizes == 0):
                voxel_sizes = np.array([1.0, 1.0, 1.0])

            # Create a dummy NIfTI reference image
            nifti_header = nib.Nifti1Header()
            nifti_header.set_data_shape(dimensions)
            nifti_header.set_zooms(voxel_sizes)
            nifti_header.set_qform(affine)
            nifti_header.set_sform(affine)

            dummy_data = np.empty(dimensions, dtype=np.int8)
            reference_img = nib.Nifti1Image(dummy_data, affine, header=nifti_header)

            # Create tractogram and TRX object
            tractogram = nib.streamlines.Tractogram(
                streamlines=streamlines, affine_to_rasmm=np.eye(4)
            )
            trx_obj = tbx.TrxFile.from_lazy_tractogram(tractogram, reference_img)

            # Save TRX file (no header argument)
            tbx.save(trx_obj, output_path)

        elif output_ext in (".vtk", ".vtp"):
            import vtk
            from vtk.util import numpy_support

            # Create VTK polydata
            vtk_points = vtk.vtkPoints()
            vtk_lines = vtk.vtkCellArray()

            point_id = 0
            for sl in streamlines:
                n_pts = len(sl)
                vtk_lines.InsertNextCell(n_pts)
                for pt in sl:
                    vtk_points.InsertNextPoint(pt[0], pt[1], pt[2])
                    vtk_lines.InsertCellPoint(point_id)
                    point_id += 1

            polydata = vtk.vtkPolyData()
            polydata.SetPoints(vtk_points)
            polydata.SetLines(vtk_lines)

            if output_ext == ".vtk":
                writer = vtk.vtkPolyDataWriter()
                writer.SetFileTypeToASCII()
            else:
                writer = vtk.vtkXMLPolyDataWriter()

            writer.SetFileName(output_path)
            writer.SetInputData(polydata)
            writer.Write()

        # Restore original logging level
        root_logger.setLevel(original_level)

        print(f"Successfully saved: {output_path}")
        logger.info(f"Conversion complete: {output_path}")

    except Exception as e:
        # Restore logging level on error
        try:
            root_logger.setLevel(original_level)
        except NameError:
            pass
        logger.error(f"Conversion failed: {e}", exc_info=True)
        print(f"Error: Conversion failed: {e}", file=sys.stderr)
        sys.exit(1)


def _run_headless_density_map(
    input_path: str, output_path: str, anat_path: str = None
) -> None:
    """
    Computes and saves a track density imaging (TDI) map without GUI.

    Args:
        input_path: Path to the input bundle file.
        output_path: Path to the output NIfTI file (.nii.gz or .nii).
        anat_path: Optional path to anatomical image for grid alignment.

    Raises:
        SystemExit: If input is missing or computation fails.
    """
    import os
    import numpy as np

    # Validate arguments
    if not input_path:
        logger.error("Error: Input bundle file is required for --density-map.")
        print(
            "Error: Input bundle file is required for --density-map.", file=sys.stderr
        )
        print("Usage: tractedit input.trk --density-map output.nii.gz", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(input_path):
        logger.error(f"Error: Input file not found: {input_path}")
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Validate output extension
    if not (output_path.endswith(".nii.gz") or output_path.endswith(".nii")):
        logger.error("Error: Output file must be .nii.gz or .nii format.")
        print("Error: Output file must be .nii.gz or .nii format.", file=sys.stderr)
        sys.exit(1)

    print(f"Computing density map: {input_path} -> {output_path}")
    logger.info(f"Headless density map: {input_path} -> {output_path}")

    try:
        import nibabel as nib
        import trx.trx_file_memmap as tbx

        # Load streamlines
        input_ext = os.path.splitext(input_path)[1].lower()
        streamlines = None
        affine = None
        shape = None

        if input_ext == ".trk":
            trk = nib.streamlines.load(input_path)
            streamlines = list(trk.streamlines)
            if hasattr(trk, "affine"):
                affine = trk.affine
            if hasattr(trk.header, "get") and "dimensions" in dict(trk.header):
                shape = tuple(int(d) for d in trk.header["dimensions"][:3])

        elif input_ext == ".tck":
            tck = nib.streamlines.load(input_path)
            streamlines = list(tck.streamlines)

        elif input_ext == ".trx":
            trx_file = tbx.load(input_path)
            streamlines = list(trx_file.streamlines)
            if "VOXEL_TO_RASMM" in trx_file.header:
                affine = trx_file.header["VOXEL_TO_RASMM"]
            if "DIMENSIONS" in trx_file.header:
                shape = tuple(int(d) for d in trx_file.header["DIMENSIONS"][:3])

        elif input_ext in (".vtk", ".vtp"):
            import vtk
            from vtk.util import numpy_support

            if input_ext == ".vtk":
                reader = vtk.vtkPolyDataReader()
            else:
                reader = vtk.vtkXMLPolyDataReader()

            reader.SetFileName(input_path)
            reader.Update()
            polydata = reader.GetOutput()

            lines = polydata.GetLines()
            points = numpy_support.vtk_to_numpy(polydata.GetPoints().GetData())

            streamlines = []
            lines.InitTraversal()
            id_list = vtk.vtkIdList()
            while lines.GetNextCell(id_list):
                n_pts = id_list.GetNumberOfIds()
                sl = np.zeros((n_pts, 3))
                for i in range(n_pts):
                    sl[i] = points[id_list.GetId(i)]
                streamlines.append(sl)
        else:
            logger.error(f"Error: Unsupported input format: {input_ext}")
            print(f"Error: Unsupported input format: {input_ext}", file=sys.stderr)
            sys.exit(1)

        if not streamlines:
            raise ValueError("No streamlines loaded from input file.")

        print(f"Loaded {len(streamlines)} streamlines.")

        # Determine grid (affine and shape)
        # Priority A: Use anatomical image if provided
        if anat_path and os.path.isfile(anat_path):
            print(f"Using anatomical reference: {anat_path}")
            anat_img = nib.load(anat_path)
            affine = anat_img.affine
            shape = anat_img.shape[:3]

        # Priority B: Use header info from bundle
        elif affine is not None and shape is not None:
            print("Using grid from bundle header.")

        # Priority C: Compute from streamline bounds
        else:
            print("Computing grid from streamline bounds...")
            all_points = np.concatenate(streamlines, axis=0)
            min_coord = np.min(all_points, axis=0)
            max_coord = np.max(all_points, axis=0)

            # 1mm isotropic with 5mm padding
            voxel_size = np.array([1.0, 1.0, 1.0])
            padding = 5.0
            min_coord -= padding
            max_coord += padding

            dims = np.ceil((max_coord - min_coord) / voxel_size).astype(int)
            shape = tuple(dims)

            affine = np.eye(4)
            affine[:3, :3] = np.diag(voxel_size)
            affine[:3, 3] = min_coord

        # Compute density map
        print("Computing density...")
        inv_affine = np.linalg.inv(affine)

        # Flatten all streamline points
        all_points = np.concatenate(streamlines, axis=0)

        # Transform to voxel coordinates
        vox_coords = nib.affines.apply_affine(inv_affine, all_points)
        vox_indices = np.rint(vox_coords).astype(int)

        # Filter points outside grid
        valid_mask = (
            (vox_indices[:, 0] >= 0)
            & (vox_indices[:, 0] < shape[0])
            & (vox_indices[:, 1] >= 0)
            & (vox_indices[:, 1] < shape[1])
            & (vox_indices[:, 2] >= 0)
            & (vox_indices[:, 2] < shape[2])
        )
        valid_voxels = vox_indices[valid_mask]

        # Create density array
        density_data = np.zeros(shape, dtype=np.int32)
        np.add.at(
            density_data,
            (valid_voxels[:, 0], valid_voxels[:, 1], valid_voxels[:, 2]),
            1,
        )

        # Save NIfTI
        nifti_img = nib.Nifti1Image(density_data.astype(np.float32), affine)

        # Copy header info from anatomical if available
        if anat_path and os.path.isfile(anat_path):
            try:
                ref_img = nib.load(anat_path)
                nifti_img.header.set_zooms(ref_img.header.get_zooms()[:3])
                nifti_img.header.set_xyzt_units(*ref_img.header.get_xyzt_units())
            except Exception:
                pass

        nib.save(nifti_img, output_path)

        max_density = np.max(density_data)
        print(f"Successfully saved: {output_path}")
        print(f"Grid shape: {shape}, Max density: {max_density}")
        logger.info(f"Density map saved: {output_path}")

    except Exception as e:
        logger.error(f"Density map failed: {e}", exc_info=True)
        print(f"Error: Density map failed: {e}", file=sys.stderr)
        sys.exit(1)


# ============================================================================
# Main Entry Point
# ============================================================================


def main() -> None:
    """
    Main function to start the tractedit application.


    """
    # Parse Command Line Arguments
    parser = argparse.ArgumentParser(description="TractEdit - GUI")
    parser.add_argument(
        "bundle",
        nargs="?",
        help="Path to the bundle file (.trk, .tck, .trx, .vtk, .vtp)",
    )
    parser.add_argument("--anat", help="Path to the anatomical image (T1w)")
    parser.add_argument(
        "--load-roi",
        action="append",
        dest="roi_paths",
        help="Path to ROI image(s). Can be used multiple times.",
    )
    parser.add_argument(
        "--roi",
        nargs=3,
        type=float,
        action="append",
        metavar=("X", "Y", "Z"),
        help="Create a sphere ROI at these coordinates (X Y Z). Can be used multiple times.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        action="append",
        help="Radius of the sphere ROI (default: 5mm). Can be used multiple times.",
    )
    parser.add_argument(
        "--convert-to",
        dest="convert_to",
        metavar="OUTPUT",
        help="Headless conversion: convert bundle to OUTPUT file format without GUI. "
        "Supported formats: .trk, .tck, .trx, .vtk, .vtp",
    )
    parser.add_argument(
        "--density-map",
        dest="density_map",
        metavar="OUTPUT",
        help="Headless export: compute and save density map as NIfTI (.nii.gz) without GUI. "
        "Use --anat to align to anatomical image grid.",
    )
    args = parser.parse_args()

    # Handle headless conversion mode
    if args.convert_to:
        _run_headless_conversion(args.bundle, args.convert_to)
        sys.exit(0)

    # Handle headless density map export
    if args.density_map:
        _run_headless_density_map(args.bundle, args.density_map, args.anat)
        sys.exit(0)

    # Windows: Set app ID for taskbar grouping
    myappid = "tractedit.app.gui"
    if sys.platform == "win32":
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        except Exception as e:
            logger.warning(f"Warning: Could not set AppUserModelID: {e}")

    # Linux: Set argv[0] for X11 WM_CLASS matching
    if sys.platform.startswith("linux"):
        if sys.argv:
            sys.argv[0] = "tractedit"

    app: QApplication = QApplication(sys.argv)

    # Linux: Set app identifiers for taskbar icon matching
    if sys.platform.startswith("linux"):
        app.setDesktopFileName("tractedit")
        app.setApplicationName("tractedit")
        app.setApplicationDisplayName("TractEdit")

    # Splash screen and app icon
    splash = None
    try:
        logo_ref = importlib.resources.files("tractedit_pkg.assets").joinpath(
            "logo.png"
        )
        with importlib.resources.as_file(logo_ref) as logo_path:
            if logo_path.is_file():
                pixmap = QPixmap(str(logo_path))
                pixmap = pixmap.scaled(
                    400,
                    400,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                splash = LoadingSplash(pixmap)
                splash.show()

        # Load app icon for window/taskbar
        icon_ref = importlib.resources.files("tractedit_pkg.assets").joinpath(
            "tractedit.png"
        )
        with importlib.resources.as_file(icon_ref) as icon_path:
            if icon_path.is_file():
                app_icon = QIcon()
                icon_pixmap = QPixmap(str(icon_path))
                for size in [16, 24, 32, 48, 64, 128, 256]:
                    scaled = icon_pixmap.scaled(
                        size,
                        size,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                    app_icon.addPixmap(scaled)
                app.setWindowIcon(app_icon)
                logger.info(f"Loaded app icon from: {icon_path}")

    except Exception as e:
        logger.warning(f"Could not load splash screen assets: {e}")

    # VTK (goes first)
    if splash:
        splash.set_progress(10, "Initializing VTK System...")

    try:
        import vtk

        out_window: vtk.vtkOutputWindow = vtk.vtkOutputWindow()
        vtk.vtkOutputWindow.SetInstance(out_window)
        vtk.vtkObject.GlobalWarningDisplayOff()
    except ImportError:
        logger.warning("Warning: VTK not found.")
    except Exception as e:
        logger.warning(f"Warning: Error suppressing VTK output: {e}")

    # Heavy imports
    if splash:
        splash.set_progress(30, "Loading Libraries...")

    try:
        from tractedit_pkg.main_window import MainWindow
    except ImportError as e:
        logger.error(f"Error importing necessary modules: {e}")
        sys.exit(1)

    # Pre-compile Numba functions (while splash is visible)
    if splash:
        splash.set_progress(50, "Optimizing performance...")

    try:
        from tractedit_pkg.file_io import warmup_numba_functions

        warmup_numba_functions()
    except Exception as e:
        logger.warning(f"Numba warmup failed (non-critical): {e}")

    # Pre-import heavy libraries to speed up first bundle load
    if splash:
        splash.set_progress(60, "Pre-loading streamline libraries...")

    try:
        import nibabel.streamlines
        import nibabel.streamlines.array_sequence
        import nibabel.streamlines.tractogram
        import nibabel.streamlines.trk
        import nibabel.streamlines.tck
        import trx.trx_file_memmap

        logger.debug("Streamline libraries pre-loaded")
    except Exception as e:
        logger.warning(f"Library pre-load failed (non-critical): {e}")

    # Warmup selection numba functions (sphere/box selection)
    if splash:
        splash.set_progress(65, "Optimizing selection tools...")

    try:
        from tractedit_pkg.visualization.selection import (
            warmup_selection_numba_functions,
        )

        warmup_selection_numba_functions()
    except Exception as e:
        logger.warning(f"Selection warmup failed (non-critical): {e}")

    # Main Window
    if splash:
        splash.set_progress(70, "Building User Interface...")

    main_window: MainWindow = MainWindow()

    if splash:
        splash.set_progress(90, "Starting...")

    # Launch - cross-platform window maximization
    screen = app.primaryScreen()
    if screen:
        available_geometry = screen.availableGeometry()
        main_window.setGeometry(available_geometry)

    main_window.show()

    # Delay maximize to ensure window manager has fully initialized the window
    def ensure_maximized():
        main_window.showMaximized()

    QTimer.singleShot(50, ensure_maximized)

    if splash:
        splash.finish(main_window)

    # Trigger initial file loading if arguments provided
    if args.bundle or args.anat or args.roi_paths or args.roi:
        # QTimer to allow the event loop to start first
        QTimer.singleShot(
            100,
            lambda: main_window.load_initial_files(
                args.bundle, args.anat, args.roi_paths, args.roi, args.radius
            ),
        )

    logger.info("App started.")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
