import os
import numpy as np
import nibabel as nib
import trx.trx_file_memmap as tbx
import logging
from ..tractogram_metadata import add_vtk_metadata, ensure_metadata_supported
from ..reference_grid import ReferenceGrid
from ..transactional_io import transactional_save
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, Union

if TYPE_CHECKING:
    import vtk

logger = logging.getLogger(__name__)


def _save_trx_native(
    trx_source: "tbx.TrxFile",
    visible_indices: set,
    bboxes: Optional[np.ndarray],
    output_path: str,
) -> str:
    """
    Save a TRX file using the native trx-python API.

    Uses ``TrxFile.select(copy_safe=True)`` to extract visible streamlines
    without Python-level iteration, retains intersecting source groups, then
    embeds cached bounding boxes as ``data_per_streamline`` before saving.

    Parameters
    ----------
    trx_source : TrxFile
        The original loaded TRX object (memmap-backed).
    visible_indices : set of int
        Indices of streamlines to include in the saved file.
    bboxes : ndarray of shape ``(N_total, 2, 3)`` or ``None``
        Full bounding box array for all streamlines.  Only the rows
        corresponding to *visible_indices* will be embedded.
    output_path : str
        Destination file path.

    Returns
    -------
    str
        Success message.

    Raises
    ------
    Exception
        Any failure is propagated to the caller so that
        ``save_streamlines_file`` can fall back to the generic path.
    """
    from .bbox_cache import _discard_cached_bboxes, _embed_cached_bboxes

    indices_arr = np.sort(
        np.fromiter(visible_indices, dtype=np.int64, count=len(visible_indices))
    )
    logger.info(
        "TRX native save: selecting %d / %d streamlines.",
        len(indices_arr),
        len(trx_source.streamlines),
    )

    keep_groups = bool(getattr(trx_source, "groups", None))
    trx_subset = None
    try:
        trx_subset = trx_source.select(
            indices_arr, keep_group=keep_groups, copy_safe=True
        )

        if bboxes is not None:
            try:
                saved_bboxes = bboxes[indices_arr]
                _embed_cached_bboxes(trx_subset, saved_bboxes)
            except (ValueError, IndexError, TypeError) as exc:
                logger.warning(
                    "TRX native save: failed to embed bboxes (%s). "
                    "Saving continues without bbox cache.",
                    exc,
                )
        else:
            _discard_cached_bboxes(trx_subset)

        transactional_save(
            output_path,
            lambda staged_path: tbx.save(trx_subset, staged_path),
        )
    finally:
        if trx_subset is not None:
            close = getattr(trx_subset, "close", None)
            if callable(close):
                close()
    logger.info(
        "File saved successfully (TRX, native path): %s",
        os.path.basename(output_path),
    )
    return f"File saved successfully (TRX): {os.path.basename(output_path)}"


# ============================================================================
# Helper Functions
# ============================================================================


def parse_numeric_tuple_from_string(
    input_value: Union[str, List, Tuple, np.ndarray, Any],
    target_type: Type = float,
    expected_length: Optional[Union[int, Tuple[int, ...]]] = None,
) -> Any:
    """
    Parses an input (string, list, tuple, or ndarray) into a tuple/array of a specific numeric type.

    If parsing or validation fails, the original input_value is returned.

    Args:
        input_value: The input data to parse. Can be a string like "(1, 2, 3)" or "1 2 3",
                     a sequence, or a numpy array.
        target_type: The desired type for the elements (default: float).
        expected_length: The expected length (int) or shape (tuple) of the result.

    Returns:
        The parsed tuple/array cast to target_type, or input_value if parsing/validation fails.
    """

    # 1. Handle non-string inputs (Already lists, tuples, or arrays)
    if not isinstance(input_value, str):
        return _process_existing_sequence(input_value, target_type, expected_length)

    # 2. Parse by stripping brackets and splitting on commas/whitespace.
    # This handles formats like "(1, 2)", "[1, 2]", "1 2 3", "1, 2, 3"
    # without using ast.literal_eval
    cleaned_str = input_value.translate(str.maketrans("", "", "[]()"))
    parts = cleaned_str.replace(",", " ").split()

    if not parts:
        return input_value

    # Helper: parse a single token to the target type.
    # When target_type is int, parse as float first to handle "1.9" -> 1.
    def _convert(token: str) -> target_type:
        if target_type is int:
            return int(float(token))
        return target_type(token)

    # Single scalar value
    if len(parts) == 1:
        try:
            val = _convert(parts[0])
            if expected_length == 1:
                return (val,)
            return val
        except (ValueError, TypeError):
            return input_value

    # Multiple values — convert to tuple
    try:
        converted = tuple(_convert(p) for p in parts)
        return _validate_length(converted, expected_length, input_value)
    except (ValueError, TypeError):
        return input_value


def _process_existing_sequence(data: Any, dtype: Type, length_req: Any) -> Any:
    """Handles inputs that are already lists, tuples, or numpy arrays."""
    if isinstance(data, (list, tuple)):
        try:
            converted = tuple(dtype(x) for x in data)
            return _validate_length(converted, length_req, data)
        except (ValueError, TypeError):
            return data

    if isinstance(data, np.ndarray):
        try:
            # Check shape/length before casting
            if not _check_numpy_shape(data, length_req):
                return data
            return data.astype(dtype)
        except (ValueError, TypeError):
            return data

    return data


def _validate_length(
    data: tuple, expected: Optional[Union[int, Tuple[int, ...]]], original: Any
) -> Any:
    """Validates that the data tuple matches the expected length."""
    if expected is None:
        return data

    # If expected is a tuple (usually for numpy shapes), we only check dimension 0 here for tuples
    if isinstance(expected, tuple):
        return data if len(data) == expected[0] else original

    return data if len(data) == expected else original


def _check_numpy_shape(
    arr: np.ndarray, expected: Optional[Union[int, Tuple[int, ...]]]
) -> bool:
    """Validates numpy array shape or length."""
    if expected is None:
        return True
    if isinstance(expected, tuple):
        return arr.shape == expected
    return arr.ndim == 1 and len(arr) == expected


def _main_window_reference_grid(main_window: Any) -> Optional[ReferenceGrid]:
    reference_grid = getattr(main_window, "anatomical_reference_grid", None)
    if reference_grid is not None:
        return reference_grid
    reference_grid = getattr(main_window, "tractogram_reference_grid", None)
    if reference_grid is not None:
        return reference_grid
    bboxes = getattr(main_window, "streamline_bboxes", None)
    if bboxes is None or len(bboxes) == 0:
        return None
    return ReferenceGrid.from_bounds(
        np.min(bboxes[:, 0], axis=0),
        np.max(bboxes[:, 1], axis=0),
        affine=getattr(main_window, "original_trk_affine", None),
        provenance="synthetic:bundle-bounds",
    )


def _header_with_reference_grid(base_header, reference_grid):
    header = base_header.copy() if base_header is not None else {}
    selected_grid = ReferenceGrid.from_header(header, provenance="tractogram-header")
    if selected_grid is None:
        selected_grid = reference_grid
    if selected_grid is not None:
        header.update(selected_grid.header_fields())
    return header


def _prepare_trk_header(
    base_header: Dict[str, Any],
    nb_streamlines: int,
    anatomical_img_affine: Optional[np.ndarray] = None,
    reference_grid: Optional[ReferenceGrid] = None,
) -> Dict[str, Any]:
    """
    Prepares and validates the header dictionary for TRK saving.
    If voxel_order is missing in base_header, attempts to derive it from
    anatomical_img_affine, otherwise defaults to 'RAS'.
    """
    header = _header_with_reference_grid(base_header, reference_grid)
    logger.info("Preparing TRK header for saving...")
    if anatomical_img_affine is not None:
        logger.debug(f"Anatomical affine provided. Type: {type(anatomical_img_affine)}")

    # Voxel Order Logic
    raw_voxel_order_from_trk = header.get("voxel_order")
    processed_voxel_order_from_trk = None

    if isinstance(raw_voxel_order_from_trk, bytes):
        try:
            processed_voxel_order_from_trk = raw_voxel_order_from_trk.decode(
                "utf-8", errors="strict"
            )
            logger.debug(f"Decoded 'voxel_order': '{processed_voxel_order_from_trk}'")
        except UnicodeDecodeError:
            logger.warning(
                f"'voxel_order' field in TRK header (bytes: {raw_voxel_order_from_trk}) could not be decoded."
            )
    elif isinstance(raw_voxel_order_from_trk, str):
        processed_voxel_order_from_trk = raw_voxel_order_from_trk

    is_valid_trk_voxel_order = (
        isinstance(processed_voxel_order_from_trk, str)
        and len(processed_voxel_order_from_trk) == 3
    )

    if is_valid_trk_voxel_order:
        header["voxel_order"] = processed_voxel_order_from_trk.upper()
        logger.info(
            f"      - Info: Using existing 'voxel_order' from TRK header: {header['voxel_order']}."
        )
    else:
        if raw_voxel_order_from_trk is not None:
            logger.warning(
                f"      - Warning: 'voxel_order' from TRK header ('{raw_voxel_order_from_trk}') is invalid or in an unexpected format."
            )
        else:
            logger.warning(f"      - Info: 'voxel_order' missing in TRK header.")

        derived_from_anat = False
        if (
            anatomical_img_affine is not None
            and isinstance(anatomical_img_affine, np.ndarray)
            and anatomical_img_affine.shape == (4, 4)
        ):
            try:
                axcodes = nib.aff2axcodes(anatomical_img_affine)
                derived_vo_str = "".join(axcodes).upper()
                if len(derived_vo_str) == 3:
                    header["voxel_order"] = derived_vo_str
                    derived_from_anat = True
                    logger.info(
                        f"      - Info: Derived 'voxel_order' from loaded anatomical image: {header['voxel_order']}."
                    )
                else:
                    logger.warning(
                        f"      - Warning: Could not derive a valid 3-character 'voxel_order' from anatomical image affine (got: '{derived_vo_str}')."
                    )
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"      - Warning: Error deriving 'voxel_order' from anatomical image: {e}"
                )

        if not derived_from_anat:
            header["voxel_order"] = "RAS"
            logger.info(
                f"      - Info: Defaulting 'voxel_order' to 'RAS' (Standard fallback)."
            )

    # If converting from TCK/VTK -> TRK, we might lack voxel_to_rasmm
    # If an anatomical image is available, use its affine
    if "voxel_to_rasmm" not in header or header["voxel_to_rasmm"] is None:
        if anatomical_img_affine is not None:
            header["voxel_to_rasmm"] = anatomical_img_affine
            logger.info(
                "      - Info: Populated 'voxel_to_rasmm' from anatomical image."
            )
        else:
            header["voxel_to_rasmm"] = np.eye(4)
            logger.warning(
                "      - Warning: 'voxel_to_rasmm' missing and no anatomical image. Using Identity."
            )

    # Process other specific TRK header fields
    keys_to_process = {
        "voxel_sizes": {"type": float, "length": 3, "default": (1.0, 1.0, 1.0)},
        "dimensions": {
            "type": int,
            "length": 3,
            "default": (1, 1, 1),
        },  # Small valid default
        "voxel_to_rasmm": {
            "type": float,
            "shape": (4, 4),
            "default": np.identity(4, dtype=np.float32),
        },
    }

    for key, K_props in keys_to_process.items():
        original_value = header.get(key)
        processed_value = original_value
        expected_item_type = K_props["type"]
        is_matrix = "shape" in K_props

        # Decode if bytes
        if isinstance(processed_value, bytes):
            try:
                processed_value = processed_value.decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                logger.warning(
                    f"      - Warning: Could not decode bytes for '{key}'. Original value: {original_value}"
                )
                header[key] = K_props["default"]
                logger.warning(f"      - Info: Set '{key}' to default: {header[key]}")
                continue

        # Parse if string, or use if already suitable type
        if isinstance(processed_value, str):
            parsed_val = parse_numeric_tuple_from_string(
                processed_value,
                expected_item_type,
                K_props.get("length") or K_props.get("shape"),
            )
            # Check if parse_numeric_tuple_from_string returned the original string (failure)
            if not (isinstance(parsed_val, str) and parsed_val == processed_value):
                processed_value = parsed_val
            else:
                logger.info(
                    f"      - Info: Could not parse string '{processed_value}' for '{key}'."
                )

        # Validate and set
        valid_structure = False
        final_value = None

        try:
            if is_matrix:  # voxel_to_rasmm
                if (
                    isinstance(processed_value, np.ndarray)
                    and processed_value.shape == K_props["shape"]
                ):
                    final_value = processed_value.astype(expected_item_type)
                    valid_structure = True
            else:  # voxel_sizes, dimensions (tuples)
                if (
                    isinstance(processed_value, tuple)
                    and len(processed_value) == K_props["length"]
                ):
                    final_value = tuple(expected_item_type(x) for x in processed_value)
                    valid_structure = True
                elif (
                    isinstance(processed_value, np.ndarray)
                    and processed_value.ndim == 1
                    and len(processed_value) == K_props["length"]
                ):
                    final_value = tuple(processed_value.astype(expected_item_type))
                    valid_structure = True
        except (ValueError, TypeError) as e:  # Catch errors from type conversion
            logger.warning(
                f"      - Warning: Type conversion error for '{key}' (value: '{processed_value}'): {e}"
            )
            valid_structure = False

        if valid_structure:
            header[key] = final_value
        else:
            # If missing, try to fill from anatomical image if available
            if anatomical_img_affine is not None:
                if key == "voxel_sizes":
                    header[key] = tuple(nib.affines.voxel_sizes(anatomical_img_affine))
                    logger.info(f"      - Info: Derived '{key}' from anatomical image.")
                    continue
                ## TODO - handle dimensions

            logger.warning(
                f"      - Warning: '{key}' ('{original_value}') was invalid, missing, or failed processing. Defaulted to {K_props['default']}."
            )
            header[key] = K_props["default"]

    header["nb_streamlines"] = nb_streamlines
    header["voxel_order"] = header["voxel_order"].upper()

    return header


def _prepare_tck_header(
    base_header: Optional[Dict[str, Any]], nb_streamlines: int
) -> Dict[str, Any]:
    """Prepares the header dictionary for TCK saving."""
    # Start with a clean header to avoid TRK-specific fields polluting TCK
    header = {}
    header["count"] = str(nb_streamlines)

    if base_header:
        # TCK headers are flexible key-value pairs.
        # Some TRK metadata are preserved as strings.
        keys_to_preserve = ["voxel_order", "dimensions", "voxel_sizes"]

        for key in keys_to_preserve:
            if key in base_header:
                val = base_header[key]
                # Convert complex types to string representation
                if isinstance(val, (tuple, list, np.ndarray)):
                    val_str = " ".join(map(str, np.array(val).flatten()))
                    header[key] = val_str
                elif isinstance(val, bytes):
                    try:
                        header[key] = val.decode("utf-8", errors="replace")
                    except (UnicodeDecodeError, ValueError):
                        logger.debug("Failed to decode header field '%s'.", key)
                        header[key] = str(val)
                else:
                    header[key] = str(val)

    return header


def _header_triplet(value, dtype):
    if isinstance(value, str):
        parsed = parse_numeric_tuple_from_string(value, dtype, 3)
        return parsed if isinstance(parsed, tuple) and len(parsed) == 3 else None
    if isinstance(value, np.ndarray) and value.size == 3:
        value = value.flatten()
    elif not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        return tuple(dtype(item) for item in value)
    except (ValueError, TypeError):
        return None


def _prepare_trx_header(
    base_header: Optional[Dict[str, Any]],
    nb_streamlines: int,
    anatomical_img_affine: Optional[np.ndarray] = None,
    reference_grid: Optional[ReferenceGrid] = None,
) -> Dict[str, Any]:
    """
    Prepares the header dictionary for TRX saving.
    Ensures essential reference fields (affine, dimensions) are present.
    """
    header = _header_with_reference_grid(base_header, reference_grid)
    header["nb_streamlines"] = nb_streamlines

    # Clean up TCK specific
    header.pop("count", None)

    # Ensure voxel_to_rasmm exists
    if "voxel_to_rasmm" not in header or header["voxel_to_rasmm"] is None:
        if anatomical_img_affine is not None:
            header["voxel_to_rasmm"] = anatomical_img_affine
        else:
            header["voxel_to_rasmm"] = np.eye(4)

    valid_dims = _header_triplet(header.get("dimensions"), int)
    if valid_dims is None:
        header["dimensions"] = (1, 1, 1)
        logger.warning(
            "TRX header: 'dimensions' invalid or missing, defaulting to (1,1,1)"
        )
    else:
        header["dimensions"] = valid_dims

    valid_vox_sizes = _header_triplet(header.get("voxel_sizes"), float)
    if valid_vox_sizes is None:
        if anatomical_img_affine is not None:
            try:
                header["voxel_sizes"] = tuple(
                    nib.affines.voxel_sizes(anatomical_img_affine)
                )
            except (ValueError, np.linalg.LinAlgError):
                logger.debug(
                    "Failed to compute voxel sizes from affine, using default."
                )
                header["voxel_sizes"] = (1.0, 1.0, 1.0)
        else:
            header["voxel_sizes"] = (1.0, 1.0, 1.0)
    else:
        header["voxel_sizes"] = valid_vox_sizes

    # Ensure voxel_order exists
    if "voxel_order" not in header:
        header["voxel_order"] = "RAS"

    return header


def _create_vtk_polydata_from_tractogram(
    tractogram: nib.streamlines.Tractogram, *, float_points: bool = False,
) -> "vtk.vtkPolyData":
    """
    Converts a Nibabel Tractogram to vtkPolyData (lines).
    Assumes streamlines are in RASMM (world space).
    """
    import vtk
    from vtk.util import numpy_support

    poly_data = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()

    # Flatten points
    if hasattr(tractogram.streamlines, "_data"):
        # Fast path if ArraySequence
        all_points = tractogram.streamlines._data
        offsets = tractogram.streamlines._offsets
        lengths = tractogram.streamlines._lengths
    else:
        # Slow path
        all_points = np.concatenate(tractogram.streamlines)
        lengths = [len(s) for s in tractogram.streamlines]
        offsets = np.concatenate(([0], np.cumsum(lengths)[:-1]))

    # Set Points
    if float_points:
        all_points = all_points.astype(np.float32, copy=False)
    vtk_points_array = numpy_support.numpy_to_vtk(all_points, deep=True)
    points.SetData(vtk_points_array)
    poly_data.SetPoints(points)

    # Set Lines (Connectivity)
    # VTK CellArray needs [n_pts, id0, id1..., n_pts, id0...]; in Numpy for speed
    n_streamlines = len(lengths)
    total_points = len(all_points)

    # Size of connectivity array = total_points + n_streamlines (headers)
    connectivity = np.empty(total_points + n_streamlines, dtype=np.int64)

    current_conn_idx = 0
    current_pt_idx = 0

    # Vectorized approach to build connectivity is complex, using mixed approach:
    for i in range(n_streamlines):
        l = lengths[i]
        connectivity[current_conn_idx] = l
        # Create range for points
        connectivity[current_conn_idx + 1 : current_conn_idx + 1 + l] = np.arange(
            current_pt_idx, current_pt_idx + l
        )

        current_conn_idx += l + 1
        current_pt_idx += l

    # Safe legacy approach
    cell_array = vtk.vtkCellArray()
    # Handle int64 vs int32 for VTK ID types
    if vtk.vtkIdTypeArray().GetDataTypeSize() == 4:
        connectivity = connectivity.astype(np.int32)

    vtk_ids = numpy_support.numpy_to_vtk(
        connectivity, deep=True, array_type=vtk.vtkIdTypeArray().GetDataType()
    )
    cell_array.SetCells(n_streamlines, vtk_ids)
    poly_data.SetLines(cell_array)

    add_vtk_metadata(poly_data, tractogram)

    return poly_data


def save_tractogram_sync(
    tractogram: nib.streamlines.Tractogram,
    header: Dict[str, Any],
    output_path: str,
    file_ext: str,
    bboxes_to_embed: Optional[np.ndarray] = None,
    *,
    cli_format: bool = False,
    groups=None,
    data_per_group=None,
    _transactional: bool = True,
) -> str:
    """
    Saves the tractogram using nibabel or trx-python based on the extension.

    Args:
        tractogram: The nibabel Tractogram to save.
        header: Header dictionary with format-specific metadata.
        output_path: Destination file path.
        file_ext: File extension (e.g., '.trk', '.tck', '.trx').
        bboxes_to_embed: Optional bounding boxes array of shape (N, 2, 3)
            to embed in the TRX file as cached metadata. Ignored for
            non-TRX formats.

    Returns:
        Success message string.
    """
    ensure_metadata_supported(
        file_ext,
        tractogram.data_per_point,
        tractogram.data_per_streamline,
        groups,
        data_per_group,
    )

    if _transactional:
        transactional_save(
            output_path,
            lambda staged_path: save_tractogram_sync(
                tractogram,
                header,
                staged_path,
                file_ext,
                bboxes_to_embed=bboxes_to_embed,
                cli_format=cli_format,
                groups=groups,
                data_per_group=data_per_group,
                _transactional=False,
            ),
        )
        format_name = file_ext.removeprefix(".").upper()
        return (
            f"File saved successfully ({format_name}): "
            f"{os.path.basename(output_path)}"
        )

    if file_ext == ".trk":
        trk_file = nib.streamlines.TrkFile(tractogram, header=header)
        nib.streamlines.save(trk_file, output_path)
        logger.info("File saved successfully (TRK)")
        return f"File saved successfully (TRK): {os.path.basename(output_path)}"

    elif file_ext == ".tck":
        tck_file = nib.streamlines.TckFile(tractogram, header=header)
        nib.streamlines.save(tck_file, output_path)
        logger.info("File saved successfully (TCK)")
        return f"File saved successfully (TCK): {os.path.basename(output_path)}"

    elif file_ext == ".trx":
        reference_grid = ReferenceGrid.from_header(
            header,
            provenance="trx-save-header",
        )
        if reference_grid is None:
            dimensions = header.get("dimensions", (1, 1, 1))
            affine = header.get("voxel_to_rasmm", np.eye(4))
            reference_grid = ReferenceGrid(
                affine=affine,
                shape=dimensions,
                provenance="synthetic:trx-save-fallback",
            )
        streamlines = tractogram.streamlines
        if hasattr(streamlines, "_data"):
            nb_vertices = len(streamlines._data)
        else:
            nb_vertices = sum(len(streamline) for streamline in streamlines)
        trx_reference = reference_grid.trx_reference(
            nb_vertices=nb_vertices,
            nb_streamlines=len(streamlines),
        )

        trx_obj_to_save = None
        try:
            trx_obj_to_save = tbx.TrxFile.from_lazy_tractogram(
                tractogram,
                trx_reference,
            )

            if groups:
                trx_obj_to_save.groups.update(groups)
            if data_per_group:
                trx_obj_to_save.data_per_group.update(data_per_group)
            if bboxes_to_embed is not None:
                from .bbox_cache import _embed_cached_bboxes

                _embed_cached_bboxes(trx_obj_to_save, bboxes_to_embed)
            tbx.save(trx_obj_to_save, output_path)
        finally:
            if trx_obj_to_save is not None:
                close = getattr(trx_obj_to_save, "close", None)
                if callable(close):
                    close()
        logger.info("File saved successfully (TRX)")
        return f"File saved successfully (TRX): {os.path.basename(output_path)}"

    elif file_ext in [
        ".vtk",
        ".vtp",
    ]:  # VTK format doesn't have a dedicated affine field.

        import vtk
        from ..utils import write_vtk_polydata

        # Apply affine to coordinates if not identity
        if not np.allclose(tractogram.affine_to_rasmm, np.eye(4)):
            # Nibabel apply_affine is convenient here
            streamlines_world = list(
                nib.streamlines.transform_streamlines(
                    tractogram.streamlines, tractogram.affine_to_rasmm
                )
            )
            # Create a temporary tractogram in RASMM (Identity affine) for saving
            temp_tractogram = nib.streamlines.Tractogram(
                streamlines_world,
                data_per_point=tractogram.data_per_point,
                data_per_streamline=tractogram.data_per_streamline,
                affine_to_rasmm=np.eye(4),
            )
            poly_data = _create_vtk_polydata_from_tractogram(temp_tractogram)
        else:
            poly_data = _create_vtk_polydata_from_tractogram(tractogram, float_points=cli_format)

        # Write
        if file_ext == ".vtp":
            writer = vtk.vtkXMLPolyDataWriter()
            if not cli_format:
                writer.SetDataModeToBinary()
        else:
            writer = vtk.vtkPolyDataWriter()
            if cli_format:
                writer.SetFileTypeToASCII()
            else:
                writer.SetFileTypeToBinary()

        write_vtk_polydata(writer, poly_data, output_path)

        logger.info(f"File saved successfully ({file_ext.upper()})")
        return f"File saved successfully ({file_ext.upper()}): {os.path.basename(output_path)}"

    else:
        raise ValueError(f"Unsupported save extension: {file_ext}")


_save_tractogram_file = save_tractogram_sync


def load_tractogram_sync(path, *, on_vtk_progress=None):
    """Read a native file object or VTK polydata. The caller owns a TRX result."""
    extension = os.path.splitext(path)[1].lower()
    if extension == ".trx":
        return tbx.load(path)
    if extension in (".trk", ".tck"):
        return nib.streamlines.load(path, lazy_load=False)
    if extension not in (".vtk", ".vtp"):
        raise ValueError(f"Unsupported input format: {extension}")
    import vtk

    reader = vtk.vtkPolyDataReader() if extension == ".vtk" else vtk.vtkXMLPolyDataReader()
    reader.SetFileName(path)
    observer = None
    if on_vtk_progress is not None:
        observer = reader.AddObserver("ProgressEvent", on_vtk_progress)
    try:
        reader.Update()
        return reader.GetOutput()
    finally:
        if observer is not None:
            reader.RemoveObserver(observer)
