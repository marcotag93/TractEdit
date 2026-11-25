# -*- coding: utf-8 -*-
"""Scientific surface-mapping contracts and private scanner-RAS geometry.

No GUI integration or association engine is installed by importing this module.
Published source buffers are immutable; intentional NumPy edits must call the
owning VTK array's Modified() before a geometry key can detect them.
"""

from .models import (
    ALGORITHM_VERSION,
    ENDPOINT_DTYPE,
    INTERSECTION_DTYPE,
    INTERVAL_CELL_DTYPE,
    INTERVAL_DTYPE,
    QUERY_CHUNK_ITEMS,
    EndpointStatus,
    IntersectionClass,
    SurfaceGeometry,
    SurfaceGeometryDiagnostics,
    SurfaceGeometryError,
    SurfaceGeometryKey,
    SurfaceMappingCancelled,
    SurfaceMappingDiagnostics,
    SurfaceMappingMode,
    SurfaceMappingParameters,
    SurfaceMappingRequest,
    SurfaceMappingResult,
    SurfaceMappingScope,
    _GEOMETRY_CHUNK,
    _ResultStorage,
    _check_cancel,
    _nonplanar_polygons,
    _topology_counts,
    _validate_affine,
    _validate_cells,
    _validated_affine_values,
    mapping_source,
    prepare_surface_geometry,
    surface_geometry_key,
)
