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
import pathlib

# PyInstaller freeze support
if __name__ == "__main__":
    multiprocessing.freeze_support()

# ============================================================================
# BLAS Thread Configuration
# ============================================================================
# Prevent BLAS/LAPACK libraries from spawning their own threads, which can
# conflict with the AOT ThreadPoolExecutor-based parallelism used by
# TractEdit's compiled kernels.
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

    def mousePressEvent(self, event):
        """Ignore mouse clicks to prevent splash from closing."""
        event.ignore()

    def keyPressEvent(self, event):
        """Ignore keyboard input to prevent splash from closing."""
        event.ignore()

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


def _read_vtk_tractogram_data(
    input_path: str, include_metadata: bool = True
) -> "tuple":
    """
    Reads geometry and numeric metadata from a VTK or VTP polydata file.

    Uses VTK 9.x's CSR cell-array API (GetOffsetsArray / GetConnectivityArray)
    together with numpy vectorised operations.

    Args:
        input_path: Absolute path to the .vtk or .vtp file.

    Returns:
        Streamlines, per-point data, and per-streamline data.

    Raises:
        ValueError: If the file contains no valid streamline data.
    """
    import numpy as np
    import vtk
    from vtk.util import numpy_support

    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".vtk":
        reader = vtk.vtkPolyDataReader()
    else:
        reader = vtk.vtkXMLPolyDataReader()

    reader.SetFileName(input_path)
    reader.Update()
    polydata = reader.GetOutput()

    vtk_points = polydata.GetPoints()
    if vtk_points is None:
        raise ValueError(f"No point data found in {input_path}")

    points = numpy_support.vtk_to_numpy(vtk_points.GetData()).astype(
        np.float32, copy=False
    )

    vtk_lines = polydata.GetLines()
    if vtk_lines is None or vtk_lines.GetNumberOfCells() == 0:
        raise ValueError(f"No line cells (streamlines) found in {input_path}")

    # VTK 9.x exposes the cell array as two parallel CSR arrays:
    #   offsets_arr  : (N+1,) cumulative start positions [0, len_0, len_0+len_1, ...]
    #   connectivity : (total_pts,) flat list of all point IDs in cell order
    offsets_arr = numpy_support.vtk_to_numpy(vtk_lines.GetOffsetsArray())
    connectivity_arr = numpy_support.vtk_to_numpy(vtk_lines.GetConnectivityArray())

    flat_coords = points[connectivity_arr].astype(np.float32, copy=False)

    # np.split with the interior offset positions yields per-streamline views
    # with no additional data copy.
    streamlines = np.split(flat_coords, offsets_arr[1:-1])
    if not include_metadata:
        return streamlines, {}, {}

    from tractedit_pkg.tractogram_metadata import extract_vtk_metadata

    lengths = np.diff(offsets_arr).astype(np.intp)
    data_per_point, data_per_streamline = extract_vtk_metadata(
        polydata,
        connectivity_arr,
        offsets_arr.astype(np.intp, copy=False),
        lengths,
    )
    return streamlines, data_per_point, data_per_streamline


def _read_vtk_streamlines(input_path: str) -> "list":
    """Read only streamline geometry from a VTK or VTP file."""
    streamlines, _, _ = _read_vtk_tractogram_data(
        input_path, include_metadata=False
    )
    return streamlines


def _run_headless_conversion(
    input_path: str, output_path: str, *, _announce: bool = True
) -> None:
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

    if _announce:
        print(f"Converting: {input_path} -> {output_path}")
        logger.info("Headless conversion: %s -> %s", input_path, output_path)
    same_path = (
        pathlib.Path(input_path).resolve() == pathlib.Path(output_path).resolve()
    )

    try:
        if same_path:
            import shutil
            import tempfile

            descriptor, snapshot_path = tempfile.mkstemp(
                dir=pathlib.Path(input_path).parent,
                prefix=".tractedit-source-",
                suffix=input_ext,
            )
            os.close(descriptor)
            try:
                shutil.copyfile(input_path, snapshot_path)
                _run_headless_conversion(snapshot_path, output_path, _announce=False)
            finally:
                try:
                    os.unlink(snapshot_path)
                except OSError:
                    logger.warning(
                        "Could not remove conversion source snapshot: %s",
                        snapshot_path,
                    )
                    print(
                        f"Warning: Temporary source remains at {snapshot_path}",
                        file=sys.stderr,
                    )
            print(f"Successfully saved: {output_path}")
            return

        import nibabel as nib
        from nibabel.streamlines import Field
        import trx.trx_file_memmap as tbx
        from tractedit_pkg.tractogram_metadata import (
            add_vtk_metadata,
            ensure_metadata_supported,
        )
        from tractedit_pkg.reference_grid import ReferenceGrid
        from tractedit_pkg.transactional_io import staged_output
        from tractedit_pkg.input_validation import validate_tractogram_data

        # Suppress verbose INFO logs from trx library
        root_logger = logging.getLogger()
        original_level = root_logger.level
        root_logger.setLevel(logging.WARNING)

        # Load input file
        streamlines = None
        header = None
        affine = None
        data_per_point = {}
        data_per_streamline = {}
        groups = {}
        data_per_group = {}
        trx_file = None
        trx_obj = None
        reference_grid = None

        if input_ext == ".trk":
            trk = nib.streamlines.load(input_path)
            streamlines = list(trk.streamlines)
            header = dict(trk.header)
            affine = trk.affine
            reference_grid = ReferenceGrid.from_header(
                header,
                provenance="bundle:.trk",
            )
            data_per_point = dict(trk.tractogram.data_per_point)
            data_per_streamline = dict(
                trk.tractogram.data_per_streamline
            )

        elif input_ext == ".tck":
            tck = nib.streamlines.load(input_path)
            streamlines = list(tck.streamlines)
            data_per_point = dict(tck.tractogram.data_per_point)
            data_per_streamline = dict(
                tck.tractogram.data_per_streamline
            )
            # TCK doesn't have affine, use identity
            affine = np.eye(4)

        elif input_ext == ".trx":
            trx_file = tbx.load(input_path)
            streamlines = list(trx_file.streamlines)
            affine = trx_file.header.get("VOXEL_TO_RASMM", np.eye(4))
            header = dict(trx_file.header)
            reference_grid = ReferenceGrid.from_header(
                header,
                provenance="bundle:.trx",
            )
            data_per_point = dict(trx_file.data_per_vertex)
            data_per_streamline = dict(trx_file.data_per_streamline)
            groups = dict(trx_file.groups)
            data_per_group = dict(trx_file.data_per_group)
        elif input_ext in (".vtk", ".vtp"):
            streamlines, data_per_point, data_per_streamline = (
                _read_vtk_tractogram_data(input_path)
            )
            affine = np.eye(4)

        if not streamlines:
            raise ValueError("No streamlines loaded from input file.")
        validate_tractogram_data(
            streamlines, data_per_point, data_per_streamline
        )

        print(f"Loaded {len(streamlines)} streamlines.")

        if reference_grid is None and output_ext in {".trk", ".trx"}:
            all_points = np.concatenate(streamlines)
            reference_grid = ReferenceGrid.from_points(
                all_points,
                affine=affine,
                provenance=f"synthetic:{input_ext}:bounds",
            )

        output_data_per_streamline = data_per_streamline
        if output_ext != ".trx":
            output_data_per_streamline = {
                key: values
                for key, values in data_per_streamline.items()
                if key != "_tractedit_bboxes"
            }
        ensure_metadata_supported(
            output_ext,
            data_per_point,
            output_data_per_streamline,
            groups,
            data_per_group,
        )
        tractogram = nib.streamlines.Tractogram(
            streamlines=streamlines,
            data_per_point=data_per_point or None,
            data_per_streamline=output_data_per_streamline or None,
            affine_to_rasmm=np.eye(4),
        )

        with staged_output(output_path) as staged_path:
            staged_output = str(staged_path)
            try:
                if output_ext == ".trk":
                    trk_header = {
                        Field.VOXEL_TO_RASMM: reference_grid.affine,
                        Field.DIMENSIONS: reference_grid.shape,
                        Field.VOXEL_SIZES: reference_grid.voxel_sizes,
                        Field.VOXEL_ORDER: reference_grid.voxel_order,
                    }
                    trk_file = nib.streamlines.TrkFile(
                        tractogram, header=trk_header
                    )
                    nib.streamlines.save(trk_file, staged_output)

                elif output_ext == ".tck":
                    tck_file = nib.streamlines.TckFile(tractogram)
                    nib.streamlines.save(tck_file, staged_output)

                elif output_ext == ".trx":
                    trx_reference = reference_grid.trx_reference(
                        nb_vertices=len(tractogram.streamlines._data),
                        nb_streamlines=len(tractogram.streamlines),
                    )
                    trx_obj = tbx.TrxFile.from_lazy_tractogram(
                        tractogram,
                        trx_reference,
                    )
                    trx_obj.groups.update(groups)
                    trx_obj.data_per_group.update(data_per_group)
                    tbx.save(trx_obj, staged_output)

                elif output_ext in (".vtk", ".vtp"):
                    import vtk
                    from tractedit_pkg.utils import write_vtk_polydata

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
                    add_vtk_metadata(polydata, tractogram)

                    if output_ext == ".vtk":
                        writer = vtk.vtkPolyDataWriter()
                        writer.SetFileTypeToASCII()
                    else:
                        writer = vtk.vtkXMLPolyDataWriter()

                    write_vtk_polydata(writer, polydata, staged_output)
            finally:
                for name in ("trx_obj", "trx_file"):
                    resource = locals().get(name)
                    if resource is not None:
                        resource.close()
                        if name == "trx_obj":
                            trx_obj = None
                        else:
                            trx_file = None
                resource = None

        # Restore original logging level
        root_logger.setLevel(original_level)

        if _announce:
            print(f"Successfully saved: {output_path}")
            logger.info("Conversion complete: %s", output_path)

    except Exception as e:
        # Restore logging level on error
        try:
            root_logger.setLevel(original_level)
        except NameError:
            pass
        logger.error(f"Conversion failed: {e}", exc_info=True)
        print(f"Error: Conversion failed: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        for trx_resource in (locals().get("trx_obj"), locals().get("trx_file")):
            if trx_resource is not None:
                trx_resource.close()


def _run_headless_density_map(
    input_path: str, output_path: str, anat_path: str | None = None
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

    if anat_path is not None and not os.path.isfile(anat_path):
        logger.error(f"Error: Anatomical reference not found: {anat_path}")
        print(
            f"Error: Anatomical reference not found: {anat_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Computing density map: {input_path} -> {output_path}")
    logger.info(f"Headless density map: {input_path} -> {output_path}")

    try:
        import nibabel as nib
        import trx.trx_file_memmap as tbx
        from tractedit_pkg.reference_grid import ReferenceGrid
        from tractedit_pkg.transactional_io import transactional_save
        from tractedit_pkg.input_validation import validate_tractogram_data

        # Load streamlines
        input_ext = os.path.splitext(input_path)[1].lower()
        streamlines = None
        affine = None
        shape = None
        reference_grid = None

        if input_ext == ".trk":
            trk = nib.streamlines.load(input_path)
            streamlines = list(trk.streamlines)
            if hasattr(trk, "affine"):
                affine = trk.affine
            reference_grid = ReferenceGrid.from_header(
                dict(trk.header),
                provenance="bundle:.trk",
            )

        elif input_ext == ".tck":
            tck = nib.streamlines.load(input_path)
            streamlines = list(tck.streamlines)

        elif input_ext == ".trx":
            trx_file = tbx.load(input_path)
            streamlines = list(trx_file.streamlines)
            if "VOXEL_TO_RASMM" in trx_file.header:
                affine = trx_file.header["VOXEL_TO_RASMM"]
            reference_grid = ReferenceGrid.from_header(
                trx_file.header,
                provenance="bundle:.trx",
            )

        elif input_ext in (".vtk", ".vtp"):
            streamlines = _read_vtk_streamlines(input_path)
        else:
            logger.error(f"Error: Unsupported input format: {input_ext}")
            print(f"Error: Unsupported input format: {input_ext}", file=sys.stderr)
            sys.exit(1)

        if not streamlines:
            raise ValueError("No streamlines loaded from input file.")
        # A finite one-point fiber contributes one TDI count by design.
        validate_tractogram_data(streamlines, allow_singleton=True)

        print(f"Loaded {len(streamlines)} streamlines.")

        # Determine grid (affine and shape)
        # Priority A: Use anatomical image if provided
        if anat_path and os.path.isfile(anat_path):
            print(f"Using anatomical reference: {anat_path}")
            from tractedit_pkg.reference_grid import canonicalize_nifti

            anat_img = canonicalize_nifti(nib.load(anat_path))
            reference_grid = ReferenceGrid.from_nifti(
                anat_img,
                provenance=f"anatomical:{anat_path}",
            )

        # Priority B: Use header info from bundle
        elif reference_grid is not None:
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
            reference_grid = ReferenceGrid(
                affine=affine,
                shape=shape,
                provenance=f"synthetic:{input_ext}:tdi-bounds",
            )

        affine = reference_grid.affine
        shape = reference_grid.shape

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
        nifti_img = reference_grid.create_nifti(density_data.astype(np.float32))

        transactional_save(output_path, lambda path: nib.save(nifti_img, path))

        max_density = np.max(density_data)
        print(f"Successfully saved: {output_path}")
        print(f"Grid shape: {shape}, Max density: {max_density}")
        logger.info(f"Density map saved: {output_path}")

    except Exception as e:
        logger.error(f"Density map failed: {e}", exc_info=True)
        print(f"Error: Density map failed: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        trx_resource = locals().get("trx_file")
        if trx_resource is not None:
            trx_resource.close()


# ============================================================================
# Main Entry Point
# ============================================================================


def main() -> None:
    """
    Main function to start the tractedit application.


    """
    # Application version
    try:
        from tractedit_pkg import __version__ as _version
    except Exception:
        _version = "unknown"

    # Handle version flags before argparse to ensure proper multiline output
    if any(flag in sys.argv for flag in ("--version", "-V", "-v")):
        _sep = "=" * 60
        print(_sep)
        print("TractEdit ")
        print(f"Version: {_version}")
        print("Author: Marco Tagliaferri, PhD Candidate ")
        print("Center for Mind/Brain Sciences (CIMeC), University of Trento, Italy")
        print("https://github.com/marcotag93/TractEdit")
        print(_sep)
        sys.exit(0)

    from tractedit_pkg.cli import build_argument_parser

    parser = build_argument_parser()
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

    # ------------------------------------------------------------------
    # Parallel Background Import (VTK+FURY / Nibabel+SciPy)
    # ------------------------------------------------------------------
    #   Lane 0 — VTK then FURY: FURY imports VTK internally, so they are
    #             serialized by Python's per-module import lock regardless,
    #             sharing a single thread
    #   Lane 1 — Nibabel + TRX + SciPy: independent, running them sequentially in
    #             one thread halves I/O and GIL contention.
    # ------------------------------------------------------------------
    import threading
    import time
    import numpy  # noqa: F401  — pre-warm before threads start

    _lane_events: list[threading.Event] = [threading.Event(), threading.Event()]
    _import_errors: list[Exception | None] = [None, None]

    _LANE_VTK_FURY = 0
    _LANE_NIB_SCIPY = 1

    def _import_vtk_and_fury() -> None:
        """Lane 0: VTK then FURY — sequential by dependency (FURY needs VTK)."""
        try:
            import vtk  # noqa: F811, F401
            from vtk.util import numpy_support  # noqa: F401
            import fury  # noqa: F401
            from fury import actor, window, colormap  # noqa: F401
        except Exception as exc:
            _import_errors[_LANE_VTK_FURY] = exc
        finally:
            _lane_events[_LANE_VTK_FURY].set()

    def _import_nibabel_and_scipy() -> None:
        """Lane 1: Nibabel, TRX, and SciPy — all independent of VTK/FURY."""
        try:
            import nibabel  # noqa: F401
            import nibabel.streamlines  # noqa: F401
            import nibabel.streamlines.array_sequence  # noqa: F401
            import nibabel.streamlines.tractogram  # noqa: F401
            import nibabel.streamlines.trk  # noqa: F401
            import nibabel.streamlines.tck  # noqa: F401
            import trx.trx_file_memmap  # noqa: F401
            from scipy.ndimage import gaussian_filter  # noqa: F401
            from scipy.ndimage import binary_dilation  # noqa: F401
            from scipy.special import sph_harm_y  # noqa: F401
        except Exception as exc:
            _import_errors[_LANE_NIB_SCIPY] = exc
        finally:
            _lane_events[_LANE_NIB_SCIPY].set()

    for _target in [_import_vtk_and_fury, _import_nibabel_and_scipy]:
        threading.Thread(target=_target, daemon=True).start()

    if splash:
        splash.set_progress(5, "Loading VTK / FURY / Nibabel...")

    # Wait for Lane 1 (faster) while keeping the splash responsive
    while not _lane_events[_LANE_NIB_SCIPY].is_set():
        if splash:
            QApplication.processEvents()
        time.sleep(0.05)

    if splash:
        splash.set_progress(20, "Loading VTK / FURY...")

    # Wait for Lane 0 (VTK + FURY — the long pole)
    while not _lane_events[_LANE_VTK_FURY].is_set():
        if splash:
            QApplication.processEvents()
        time.sleep(0.05)

    if splash:
        splash.set_progress(40, "Finalizing imports...")

    logger.info("Background library imports complete.")

    # QVTKRenderWindowInteractor needs both VTK and Qt
    try:
        from vtkmodules.qt.QVTKRenderWindowInteractor import (  # noqa: F401
            QVTKRenderWindowInteractor,
        )
    except Exception as exc:
        logger.warning("QVTKRenderWindowInteractor import failed: %s", exc)

    # Report any lane errors (non-fatal).
    _lane_labels = ["VTK+FURY", "Nibabel+SciPy"]
    for idx, err in enumerate(_import_errors):
        if err is not None:
            logger.warning(
                "%s background import failed: %s", _lane_labels[idx], err
            )

    # Configure VTK on the main thread
    if _import_errors[_LANE_VTK_FURY] is None:
        try:
            import vtk 

            out_window = vtk.vtkOutputWindow()
            vtk.vtkOutputWindow.SetInstance(out_window)
            vtk.vtkObject.GlobalWarningDisplayOff()
            logger.info("VTK initialized successfully.")
        except Exception as e:
            logger.warning("Error suppressing VTK output: %s", e)
    else:
        logger.warning("VTK import failed: %s", _import_errors[_LANE_VTK_FURY])

    # Main Window import
    if splash:
        splash.set_progress(50, "Loading application modules...")
    logger.info("Loading main window module...")

    try:
        from tractedit_pkg.main_window import MainWindow

        logger.info("Main window module loaded.")
    except ImportError as e:
        logger.error(f"Error importing necessary modules: {e}")
        sys.exit(1)

    # Main Window
    if splash:
        splash.set_progress(75, "Building User Interface...")
    logger.info("Building main window UI...")

    main_window: MainWindow = MainWindow()
    logger.info("Main window created.")

    if splash:
        splash.set_progress(90, "Starting...")

    # Launch - start maximized
    logger.info("Launching application window...")
    main_window.showMaximized()

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
