"""Command-line tractogram conversion and density map export."""

import logging
import os
import pathlib
import sys

logger = logging.getLogger(__name__)


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
    from vtk.util import numpy_support
    from .file_io.tractogram import load_tractogram_sync

    polydata = load_tractogram_sync(input_path)

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
        from .file_io.tractogram import load_tractogram_sync, save_tractogram_sync
        from tractedit_pkg.tractogram_metadata import (
            ensure_metadata_supported,
        )
        from tractedit_pkg.reference_grid import ReferenceGrid
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
        reference_grid = None

        if input_ext == ".trk":
            trk = load_tractogram_sync(input_path)
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
            tck = load_tractogram_sync(input_path)
            streamlines = list(tck.streamlines)
            data_per_point = dict(tck.tractogram.data_per_point)
            data_per_streamline = dict(
                tck.tractogram.data_per_streamline
            )
            # TCK doesn't have affine, use identity
            affine = np.eye(4)

        elif input_ext == ".trx":
            trx_file = load_tractogram_sync(input_path)
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

        from .transactional_io import staged_output

        with staged_output(output_path) as staged_path:
            save_tractogram_sync(
                tractogram,
                reference_grid.header_fields() if output_ext in (".trk", ".trx") else {},
                str(staged_path),
                output_ext,
                cli_format=True,
                _transactional=False,
                groups=groups,
                data_per_group=data_per_group,
            )
            if trx_file is not None:
                owner, trx_file = trx_file, None
                owner.close()

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
        trx_resource = locals().get("trx_file")
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
    if not output_path.lower().endswith((".nii.gz", ".nii")):
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
        from tractedit_pkg.reference_grid import ReferenceGrid
        from .file_io.density_map import density_image, density_grid_from_points
        from .file_io.tractogram import load_tractogram_sync
        from tractedit_pkg.input_validation import validate_tractogram_data

        # Load streamlines
        input_ext = os.path.splitext(input_path)[1].lower()
        streamlines = None
        reference_grid = None

        if input_ext == ".trk":
            trk = load_tractogram_sync(input_path)
            streamlines = list(trk.streamlines)
            reference_grid = ReferenceGrid.from_header(
                dict(trk.header),
                provenance="bundle:.trk",
            )

        elif input_ext == ".tck":
            tck = load_tractogram_sync(input_path)
            streamlines = list(tck.streamlines)

        elif input_ext == ".trx":
            trx_file = load_tractogram_sync(input_path)
            streamlines = list(trx_file.streamlines)
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

        points = None
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
            points = np.concatenate(streamlines, axis=0)
            reference_grid = density_grid_from_points(
                points, provenance=f"synthetic:{input_ext}:tdi-bounds"
            )

        print("Computing density...")
        if points is None:
            points = np.concatenate(streamlines, axis=0)
        nifti_img, counts = density_image(points, reference_grid)
        from .transactional_io import transactional_save

        transactional_save(output_path, lambda path: nib.save(nifti_img, path))
        shape = reference_grid.shape
        print(f"Successfully saved: {output_path}")
        print(f"Grid shape: {shape}, Max density: {np.max(counts)}")
        logger.info(f"Density map saved: {output_path}")

    except Exception as e:
        logger.error(f"Density map failed: {e}", exc_info=True)
        print(f"Error: Density map failed: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        trx_resource = locals().get("trx_file")
        if trx_resource is not None:
            trx_resource.close()
