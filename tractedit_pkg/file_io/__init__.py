"""File I/O entry points. Load GUI workers only when requested."""

from importlib import import_module


_MODULES = {
    "anatomical": (
        "MAX_VOXELS",
        "MMAP_SLICE_CACHE_SIZE",
        "MemoryMappedImage",
        "_canonicalize_image",
        "_maybe_downsample_image",
        "AnatomicalImageLoaderThread",
        "_update_vtk_and_ui_after_load",
        "load_roi_images",
        "gaussian_filter",
        "pyqtSignal",
        "QFileDialog",
        "QMessageBox",
        "QApplication",
        "CancellableWorkerThread",
        "choose_input",
        "ReferenceGrid",
        "_validate_volume_geometry",
        "Any",
        "List",
        "Optional",
        "Tuple",
    ),
    "bundle_stats": (
        "MEDOID_SAMPLING_THRESHOLD",
        "MEDOID_SAMPLE_SIZE",
        "_MEDOID_DISTANCE_BATCH",
        "CENTROID_MAX_STREAMLINES",
        "_legacy_medoid_sample_indices",
        "_historical_mdf_distance",
        "_historical_mdf_row_total",
        "_refine_historical_medoid_candidates",
        "_compute_medoid_totals",
        "MedoidCalculationThread",
        "_compute_centroid_math",
        "_finalize_statistic_save",
        "calculate_and_save_statistic",
        "Qt",
        "QTimer",
        "QProgressDialog",
        "Callable",
        "_resample_streamline_numba",
        "_compute_centroid_numba",
    ),
    "bundle_gui": (
        "_BUNDLE_STATE_ATTRIBUTES",
        "StreamlineLoaderThread",
        "load_streamlines_file",
        "_validate_save_prerequisites",
        "_get_save_path_and_extension",
        "_prepare_tractogram_and_affine",
        "save_streamlines_file",
        "ExitStack",
        "tbx",
        "vtk",
        "numpy_support",
        "ThemeManager",
        "ColorMode",
        "ensure_metadata_supported",
        "extract_vtk_metadata",
        "Dict",
    ),
    "bbox_cache": (
        "_packed_streamline_arrays",
        "_array_is_finite",
        "_validate_scalar_data",
        "_validate_and_compute_bboxes",
        "_TRX_BBOX_KEY",
        "_TRX_BBOX_CACHE_VERSION",
        "_TRX_BBOX_CACHE_VERSION_KEY",
        "_TRX_BBOX_CACHE_DIGEST_KEY",
        "_TRX_BBOX_GEOMETRY_DIGEST_KEY",
        "_TRX_BBOX_SAMPLE_STREAMLINES",
        "_TRX_BBOX_SAMPLE_POINTS",
        "_TRX_BBOX_HEADER_KEYS",
        "_sample_indices",
        "_bbox_cache_digest",
        "_bbox_geometry_sample_digest",
        "_discard_cached_bboxes",
        "_try_load_cached_bboxes",
        "_embed_cached_bboxes",
        "_extract_visible_bboxes",
        "hashlib",
        "zlib",
        "_validate_affine",
        "_compute_bboxes_prevalidated",
        "_validate_streamlines_numba",
    ),
    "tractogram": (
        "os",
        "np",
        "nib",
        "_save_trx_native",
        "parse_numeric_tuple_from_string",
        "_process_existing_sequence",
        "_validate_length",
        "_check_numpy_shape",
        "_main_window_reference_grid",
        "_prepare_trk_header",
        "_prepare_tck_header",
        "_prepare_trx_header",
        "_create_vtk_polydata_from_tractogram",
        "_save_tractogram_file",
        "load_tractogram_sync",
        "save_tractogram_sync",
        "add_vtk_metadata",
        "transactional_save",
        "Type",
        "Union",
    ),
    "density_map": (
        "density_image",
        "density_grid_from_points",
    ),
}
_EXTERNAL_EXPORTS = {
    "_register_background_worker": ("..workers", "register_background_worker"),
    "_worker_is_current": ("..workers", "worker_is_current"),
    "_release_background_worker": ("..workers", "release_background_worker"),
    "_release_finished_worker": ("..workers", "release_finished_worker"),
    "_release_deferred_trx_owners": ("..workers", "release_deferred_trx_owners"),
    "_retire_trx_owner": ("..workers", "retire_trx_owner"),
    "_resample_batch_numba": (".._numba_aot._parallel_wrappers", "resample_batch"),
    "_compute_bboxes_numba": (".._numba_aot._parallel_wrappers", "compute_bboxes"),
    "write_vtk_polydata": ("..utils", "write_vtk_polydata"),
}
_EXPORTS = {name: module for module, names in _MODULES.items() for name in names}
__all__ = [
    "MAX_VOXELS",
    "MMAP_SLICE_CACHE_SIZE",
    "MemoryMappedImage",
    "AnatomicalImageLoaderThread",
    "load_roi_images",
    "MEDOID_SAMPLING_THRESHOLD",
    "MEDOID_SAMPLE_SIZE",
    "CENTROID_MAX_STREAMLINES",
    "MedoidCalculationThread",
    "calculate_and_save_statistic",
    "StreamlineLoaderThread",
    "load_streamlines_file",
    "save_streamlines_file",
    "parse_numeric_tuple_from_string",
    "load_tractogram_sync",
    "save_tractogram_sync",
    "density_image",
    "density_grid_from_points",
]


def __getattr__(name):
    external = _EXTERNAL_EXPORTS.get(name)
    if external is not None:
        module_path, attr_name = external
        value = getattr(import_module(module_path, __name__), attr_name)
        globals()[name] = value
        return value
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(name)
    value = getattr(import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value
