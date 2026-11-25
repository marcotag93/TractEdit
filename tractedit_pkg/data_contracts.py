# -*- coding: utf-8 -*-

"""Typed mapping contracts shared across application managers."""

from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict

import numpy as np

from .reference_grid import ReferenceGrid

if TYPE_CHECKING:
    from vtkmodules.vtkCommonDataModel import vtkPolyData


MeshCoordinateSystem = Literal["RAS", "LPS"]
MeshCoordinateChoice = Literal["from_file", "RAS", "LPS"]
MeshMetadataSource = Literal["ply_comments", "vtp_field_data", "obj_file"]
MeshBounds = tuple[float, float, float, float, float, float]
MeshComponent = Literal[
    "surface", "edges", "vertices", "bounding_box", "boundary_edges", "surface_normals",
]


class MeshComponentPreference(TypedDict):
    enabled: bool


class MeshEdgesPreference(MeshComponentPreference):
    color: tuple[float, float, float]
    line_width: float


class MeshVerticesPreference(MeshComponentPreference):
    color: tuple[float, float, float]
    point_size: float


class MeshNormalsPreference(MeshComponentPreference):
    color: tuple[float, float, float]
    length_mm: float
    density: float


class MeshMainPresentation(TypedDict):
    surface: MeshComponentPreference
    edges: MeshEdgesPreference
    vertices: MeshVerticesPreference


class MeshPresentation(MeshMainPresentation):
    bounding_box: MeshEdgesPreference
    boundary_edges: MeshEdgesPreference
    surface_normals: MeshNormalsPreference


class MeshComponentStyle(TypedDict, total=False):
    color: tuple[float, float, float]
    opacity: float
    line_width: float
    point_size: float
    length_mm: float
    density: float


class MeshSpaceMetadata(TypedDict):
    """Conventional SPACE declarations, kept separately from the user's choice."""

    source: MeshMetadataSource
    declarations: tuple[str, ...]
    status: Literal["missing", "valid", "unknown", "conflicting", "invalid"]
    coordinate_system: MeshCoordinateSystem | None


class MeshCoordinateProvenance(TypedDict):
    choice: MeshCoordinateChoice
    metadata: MeshSpaceMetadata
    overrides_metadata: bool


class MeshCoordinates(TypedDict):
    coordinate_system: MeshCoordinateSystem
    coordinate_provenance: MeshCoordinateProvenance
    source_to_ras: np.ndarray


class ObjTextureIndex(TypedDict):
    """Lightweight OBJ material/UV summary captured during preflight without I/O."""

    reference_lines: tuple[bytes, ...]
    face_layout: dict[str, "MeshTextureFaceSummary"]
    face_uv_bytes: bytes | None


class MeshLoadResult(TypedDict):
    """Validated CPU data detached from the reader; no rendering or Qt objects."""

    path: str
    polydata: "vtkPolyData"
    space_metadata: MeshSpaceMetadata
    point_count: int
    polygon_count: int
    strip_count: int
    source_bounds: MeshBounds
    source_sha256: NotRequired[str]
    verified_geometry_sha256: NotRequired[str]
    source_stamp: NotRequired[tuple[int, ...]]
    obj_texture_index: NotRequired[ObjTextureIndex]


class SurfacePolygonRegion(TypedDict):
    reference_space: NotRequired[str]
    faces: np.ndarray
    loop: np.ndarray
    digest: str


class MeshPolyline(TypedDict):
    region: NotRequired[SurfacePolygonRegion]
    id: str
    name: str
    points: np.ndarray
    color: tuple[float, float, float]
    visible: bool
    line_width: float
    point_size: float
    show_points: bool


class MeshLayer(MeshCoordinates):
    """MainWindow-owned data; source geometry is immutable after publication."""

    id: str
    parent_id: NotRequired[str | None]
    path: str
    display_name: str
    polydata: "vtkPolyData"
    visible: bool
    opacity: float
    color: tuple[float, float, float]
    presentation: MeshPresentation
    point_count: int
    polygon_count: int
    strip_count: int
    source_bounds: MeshBounds
    source_sha256: NotRequired[str]
    source_stamp: NotRequired[tuple[int, ...]]
    obj_texture_index: NotRequired[ObjTextureIndex]
    texture: NotRequired["MeshTextureConfiguration"]
    texture_assets: NotRequired[dict[str, "MeshTextureAsset"]]
    polylines: NotRequired[list[MeshPolyline]]


class MeshTextureAsset(TypedDict):
    path: str
    size: int
    sha256: str


class MeshTextureMap(TypedDict):
    path: str
    scale: tuple[float, float, float]
    offset: tuple[float, float, float]
    clamp: bool


class MeshTextureMaterial(TypedDict):
    name: str
    library: MeshTextureAsset | None
    diffuse_map: MeshTextureMap | None
    diffuse_color: tuple[float, float, float]
    image_key: str | None
    unavailable_reason: str | None
    image_origin: NotRequired[Literal["automatic", "manual"]]
    library_origin: NotRequired[Literal["automatic", "manual"]]


class MeshTextureImage(TypedDict):
    """Cache-owned, read-only uint8 pixels in VTK's bottom-up row order."""

    asset: MeshTextureAsset
    pixels: np.ndarray
    has_transparency: bool
    resident_estimate_bytes: int
    temporary_cpu_bytes: int


class MeshTextureDiagnostic(TypedDict):
    code: str
    message: str


class MeshTextureAssetRequest(TypedDict):
    base: str
    reference: str
    kind: Literal["mtl", "image"]


class MeshTextureCosts(TypedDict):
    decoded_cpu_bytes: int
    temporary_cpu_bytes: int
    resident_estimate_bytes: int


class MeshTexturePreparation(TypedDict):
    """CPU presentation data, kept outside authoritative mesh geometry."""

    materials: dict[str, MeshTextureMaterial]
    libraries: tuple[MeshTextureAsset, ...]
    images: dict[str, MeshTextureImage]
    diagnostics: tuple[MeshTextureDiagnostic, ...]
    costs: MeshTextureCosts
    face_layout: NotRequired[dict[str, "MeshTextureFaceSummary"]]
    face_uv: NotRequired[np.ndarray]
    source: MeshTextureAsset
    unresolved_assets: NotRequired[tuple[MeshTextureAssetRequest, ...]]


class MeshTextureFaceSummary(TypedDict):
    face_count: int
    corner_count: int
    complete_uv: bool


class MeshTextureSavedMaterial(TypedDict):
    library: str | None
    image: str | None
    diffuse_map: MeshTextureMap | None
    diffuse_color: tuple[float, float, float]
    unavailable_reason: str | None
    image_origin: NotRequired[Literal["automatic", "manual"]]
    library_origin: NotRequired[Literal["automatic", "manual"]]


class MeshTextureConfiguration(TypedDict):
    enabled: bool
    libraries: tuple[str, ...]
    materials: dict[str, MeshTextureSavedMaterial]
    face_layout: dict[str, MeshTextureFaceSummary]


class StreamlineLoadResult(TypedDict):
    """Payload emitted after a successful tractogram load."""

    path: str
    ext: str
    streamlines: Any
    header: Any
    affine: np.ndarray
    scalars: dict[str, Any]
    active_scalar: str | None
    data_per_streamline: dict[str, Any]
    bboxes: np.ndarray
    reference_grid: ReferenceGrid | None
    trx_obj: NotRequired[Any]


class AnatomicalImageLoadResult(TypedDict):
    """Payload emitted after a successful anatomical image load."""

    data: np.ndarray
    affine: np.ndarray
    path: str
    was_downsampled: bool
    mmap_image: Any
    reference_grid: ReferenceGrid


class RoiLayer(TypedDict):
    """Shared state stored for one ROI layer."""

    data: np.ndarray
    affine: np.ndarray
    inv_affine: np.ndarray
    T_main_to_roi: np.ndarray
    path: NotRequired[str]
    color: NotRequired[tuple[float, float, float]]
    display_name: NotRequired[str]
