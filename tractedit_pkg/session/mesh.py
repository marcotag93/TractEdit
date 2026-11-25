"""Bounded surface and annotation records for session schemas 3 through 12."""

from __future__ import annotations

from typing import TYPE_CHECKING, NotRequired, TypedDict
from uuid import UUID

import numpy as np

from ..mesh.model import (
    MAX_MESH_LAYERS,
    create_mesh_layer, mesh_space_metadata, resolve_mesh_coordinates,
    validate_mesh_forest,
)
from ..data_contracts import (
    MeshPolyline,
    MeshCoordinateProvenance,
    MeshCoordinateSystem,
    MeshLayer,
    MeshLoadResult,
    MeshMainPresentation,
    MeshPresentation,
    MeshTextureConfiguration,
)
from ..mesh.presentation import (
    MESH_TREE_COMPONENTS, validate_mesh_presentation,
)
from .mesh_texture import (
    texture_asset_role, texture_references, validate_texture_configuration,
)

if TYPE_CHECKING:
    from vtkmodules.vtkCommonDataModel import vtkPolyData

MESH_FIELDS = (
    "id",
    "display_name",
    "coordinate_system",
    "coordinate_provenance",
    "source_to_ras",
    "visible",
    "opacity",
    "color",
)


class SessionMeshRecord(TypedDict):
    id: str
    parent_id: NotRequired[str | None]
    display_name: str
    coordinate_system: MeshCoordinateSystem
    coordinate_provenance: MeshCoordinateProvenance
    source_to_ras: np.ndarray
    visible: bool
    opacity: float
    color: tuple[float, float, float]
    texture: NotRequired[MeshTextureConfiguration]
    presentation: NotRequired[MeshMainPresentation | MeshPresentation]
    polylines: NotRequired[list[MeshPolyline]]


def _valid_flag_tuple(flags, length: int = 2) -> bool:
    return (
        isinstance(flags, tuple)
        and len(flags) == length
        and all(type(flag) is bool for flag in flags)
    )


def mesh_source_id(role: str) -> str | None:
    """Accept only canonical UUID source roles, independently of filenames."""
    if not isinstance(role, str) or not role.startswith("mesh:"):
        return None
    value = role[5:]
    try:
        return value if str(UUID(value)) == value else None
    except (ValueError, AttributeError):
        return None


def validate_mesh_records(
    records: list[SessionMeshRecord], *, legacy_polylines: bool = False,
) -> None:
    from .io import SessionError

    if not isinstance(records, list) or len(records) > MAX_MESH_LAYERS:
        raise SessionError("Invalid mesh layer count")
    identities = set()
    polyline_ids = set()
    for record in records:
        if (
            not isinstance(record, dict)
            or set(record) - {"texture", "presentation", "polylines", "parent_id"}
            != set(MESH_FIELDS)
        ):
            raise SessionError("Invalid mesh presentation fields")
        if "polylines" in record:
            from ..mesh.polylines import validate_polylines

            try:
                validate_polylines(record["polylines"], legacy=legacy_polylines)
            except (ValueError, TypeError, AttributeError) as error:
                raise SessionError("Invalid mesh polylines") from error
            for polyline in record["polylines"]:
                if polyline["id"] in polyline_ids:
                    raise SessionError("Duplicate mesh polyline identity")
                polyline_ids.add(polyline["id"])
        if "presentation" in record:
            try:
                validate_mesh_presentation(record["presentation"])
            except (ValueError, TypeError) as error:
                raise SessionError("Invalid mesh component presentation") from error
        if "texture" in record:
            validate_texture_configuration(record["texture"])
        identity = record["id"]
        if (
            not isinstance(identity, str)
            or mesh_source_id(f"mesh:{identity}") != identity
            or identity in identities
        ):
            raise SessionError("Invalid or duplicate mesh UUID")
        identities.add(identity)
        name = record["display_name"]
        if not isinstance(name, str) or not name.strip() or len(name) > 4096:
            raise SessionError("Invalid mesh display name")
        if type(record["visible"]) is not bool:
            raise SessionError("Invalid mesh visibility")
        opacity = record["opacity"]
        if (
            type(opacity) not in (int, float)
            or not np.isfinite(opacity)
            or not 0 <= opacity <= 1
        ):
            raise SessionError("Invalid mesh opacity")
        color = np.asarray(record["color"])
        if (
            color.shape != (3,)
            or color.dtype.kind not in "iuf"
            or not np.isfinite(color).all()
            or np.any((color < 0) | (color > 1))
        ):
            raise SessionError("Invalid mesh color")
        provenance = record["coordinate_provenance"]
        if not isinstance(provenance, dict) or set(provenance) != {
            "choice",
            "metadata",
            "overrides_metadata",
        }:
            raise SessionError("Invalid mesh coordinate provenance")
        metadata = provenance["metadata"]
        if not isinstance(metadata, dict) or set(metadata) != {
            "source",
            "declarations",
            "status",
            "coordinate_system",
        }:
            raise SessionError("Invalid mesh SPACE metadata")
        declarations = metadata["declarations"]
        if (
            not isinstance(declarations, tuple)
            or len(declarations) > 256
            or any(not isinstance(v, str) or len(v) > 4096 for v in declarations)
            or sum(len(v) for v in declarations) > 16384
        ):
            raise SessionError("Mesh SPACE metadata exceeds session limits")
        try:
            expected_metadata = mesh_space_metadata(
                metadata["source"],
                None if metadata["status"] == "missing" else declarations,
            )
            coordinates = resolve_mesh_coordinates(
                expected_metadata, provenance["choice"]
            )
        except (ValueError, TypeError) as error:
            raise SessionError("Invalid mesh coordinate convention") from error
        matrix = np.asarray(record["source_to_ras"])
        if (
            metadata != expected_metadata
            or type(provenance["overrides_metadata"]) is not bool
            or provenance != coordinates["coordinate_provenance"]
            or record["coordinate_system"] != coordinates["coordinate_system"]
            or matrix.shape != (4, 4)
            or matrix.dtype != np.float64
            or not np.array_equal(matrix, coordinates["source_to_ras"])
        ):
            raise SessionError("Mesh transform must match its RAS/LPS convention")
    try:
        validate_mesh_forest({record["id"]: record for record in records})
    except ValueError as error:
        raise SessionError(f"Invalid mesh hierarchy: {error}") from error


def validate_mesh_document(document: dict, schema: int | None = None) -> None:
    from .io import SessionError
    from ..mesh.polylines import POLYLINE_STYLE_FIELDS

    state, views = document.get("state"), document.get("views")
    records = state.get("mesh_layers", []) if isinstance(state, dict) else []
    polylines = [
        line for record in records if isinstance(record, dict)
        and isinstance(record.get("polylines"), list)
        for line in record["polylines"]
    ] if isinstance(records, list) else []
    has_regions = any(isinstance(line, dict) and "region" in line for line in polylines)
    if schema is not None and (schema == 12) != has_regions:
        raise SessionError("Surface polygons require schema 12 and a mesh owner")
    if isinstance(state, dict) and isinstance(state.get("standalone_polylines"), list) and any(
        isinstance(line, dict) and "region" in line for line in (state.get("standalone_polylines") or [])
    ):
        raise SessionError("Surface polygons require a mesh owner")
    legacy = schema == 8 or schema is None and not any(
        isinstance(line, dict) and set(line) & POLYLINE_STYLE_FIELDS
        for line in polylines
    ) and not (isinstance(state, dict) and state.get("standalone_polylines"))
    validate_mesh_records(records, legacy_polylines=legacy)
    validate_polyline_containers(state)
    if schema is not None:
        if schema < 10 and any("parent_id" in record for record in records):
            raise SessionError("Mesh parent fields require session schema 10")
        if schema == 10 and any("parent_id" not in record for record in records):
            raise SessionError("Missing schema 10 mesh parent field")
    if schema is not None and schema < 9 and (
        isinstance(state, dict) and "standalone_polylines" in state
        or isinstance(views, dict) and "standalone_polyline_tree" in views
    ):
        raise SessionError("Standalone polylines require session schema 9")
    if schema is not None and schema < 8 and any("polylines" in r for r in records):
        raise SessionError("Mesh polylines require session schema 8")
    roles = {role for role in document["sources"] if mesh_source_id(role) is not None}
    if roles != {"mesh:" + record["id"] for record in records}:
        raise SessionError("Mesh layers and source records must correspond exactly")
    for record in records:
        for line in record.get("polylines", ()):
            if "region" in line and (
                line["region"]["digest"] != document["sources"]["mesh:" + record["id"]]["sha256"]
            ):
                raise SessionError("Surface polygon belongs to a different source mesh")
    assets = {role for role in document["sources"] if texture_asset_role(role)}
    references = set().union(*(
        texture_references(record["texture"])
        for record in records if "texture" in record
    ))
    if assets != references:
        raise SessionError("Texture assets and layer references must correspond exactly")
    if schema in (1, 2, 3) and (assets or any("texture" in r for r in records)):
        raise SessionError("Texture fields require session schema 4")
    tree = views.get("mesh_tree") if isinstance(views, dict) else None
    component_tree = isinstance(tree, dict) and bool(
        set(tree) & {"components", "current_component"}
    )
    if schema in (1, 2, 3, 4) and (
        component_tree or any("presentation" in r for r in records)
    ):
        raise SessionError("Mesh components require session schema 5")
    if schema == 5 and (
        not component_tree or any("presentation" not in r for r in records)
    ):
        raise SessionError("Missing schema 5 mesh components")
    final_presentation = any(
        "bounding_box" in record.get("presentation", {}) for record in records
    )
    final_tree = component_tree and isinstance(tree.get("components"), dict) and any(
        isinstance(key, tuple) and len(key) == 2
        and key[1] not in ("surface", "texture", "edges", "vertices")
        for key in tree.get("components", {})
    )
    final = schema in (6, 7, 8, 9, 10, 11, 12) or schema is None and (
        final_presentation or final_tree
    )
    if schema == 5 and (final_presentation or final_tree):
        raise SessionError("Advanced and Information require session schema 6")
    if final and (records or schema in (6, 7, 8, 10)) and (
        not component_tree or any("presentation" not in record for record in records)
    ):
        raise SessionError("Missing schema 6 mesh presentation")
    for record in records:
        if "presentation" in record:
            try:
                validate_mesh_presentation(record["presentation"], advanced=final)
            except (ValueError, TypeError) as error:
                raise SessionError("Invalid mesh schema presentation") from error
    if schema in (1, 2) and (
        roles
        or (isinstance(state, dict) and "mesh_layers" in state)
        or (isinstance(views, dict) and "mesh_tree" in views)
    ):
        raise SessionError("Mesh fields require session schema 3")
    if (schema in (3, 4, 5, 6, 7, 8, 10) or records) and (
        not isinstance(state, dict) or "mesh_layers" not in state or tree is None
    ):
        raise SessionError("Missing schema 3 mesh presentation")
    if tree is not None:
        identities = {record["id"] for record in records}
        if (
            not isinstance(tree, dict)
            or set(tree) - {"annotations"} != {"items", "current", "expanded"} | (
                {"components", "current_component"} if component_tree else set()
            )
            or not isinstance(tree["items"], dict)
            or set(tree["items"]) != identities
            or (tree["current"] is not None and (
                not isinstance(tree["current"], str)
                or tree["current"] not in identities
            ))
            or type(tree["expanded"]) is not bool
            or any(
                not _valid_flag_tuple(flags)
                for flags in tree["items"].values()
            )
        ):
            raise SessionError("Invalid mesh tree presentation")
        if "annotations" in tree:
            annotations = tree["annotations"]
            trace_keys = {
                (record["id"], trace["id"])
                for record in records for trace in record.get("polylines", ())
            }
            split_keys = set(trace_keys)
            legacy_keys = set(trace_keys)
            for r in records:
                lines = r.get("polylines", ())
                if not lines:
                    continue
                legacy_keys.add((r["id"], "polylines"))
                if any("region" not in line for line in lines):
                    split_keys.add((r["id"], "polylines"))
                if any("region" in line for line in lines):
                    split_keys.add((r["id"], "polygons"))
            if (
                not isinstance(annotations, dict)
                or set(annotations) not in (split_keys, legacy_keys)
                or any(
                    not _valid_flag_tuple(flags, 3)
                    for flags in annotations.values()
                )
                or sum(flags[2] for flags in annotations.values()) > 1
            ):
                raise SessionError("Invalid mesh annotation tree presentation")
        if component_tree:
            components = tree["components"]
            current = tree["current_component"]
            expected = {
                (identity, component) for identity in identities
                for component in (
                    MESH_TREE_COMPONENTS if final else
                    ("surface", "texture", "edges", "vertices")
                )
            }
            optional = {
                (identity, component) for identity in identities
                for component in ("information_strips", "appearance")
            }
            if final and isinstance(components, dict):
                expected |= set(components) & optional
            if (
                not isinstance(components, dict)
                or set(components) != expected
                or any(
                    not _valid_flag_tuple(flags)
                    for flags in components.values()
                )
                or (current is not None and (
                    not isinstance(current, tuple) or len(current) != 2
                    or any(not isinstance(v, str) for v in current)
                    or current not in expected or tree["current"] is not None
                ))
            ):
                raise SessionError("Invalid mesh component tree presentation")
    validate_standalone_tree(state, views)


def validate_polyline_containers(state) -> None:
    from ..mesh.polylines import validate_polylines
    from .io import SessionError

    if not isinstance(state, dict):
        return
    standalone = state.get("standalone_polylines", [])
    try:
        validate_polylines(standalone)
    except (ValueError, TypeError, AttributeError) as error:
        raise SessionError("Invalid standalone polylines") from error
    identities = {record["id"] for record in standalone}
    for layer in state.get("mesh_layers", []):
        for record in layer.get("polylines", []):
            if record["id"] in identities:
                raise SessionError("Duplicate polyline identity across containers")
            identities.add(record["id"])


def validate_standalone_tree(state, views) -> None:
    from .io import SessionError

    if not isinstance(views, dict):
        return
    records = state.get("standalone_polylines", []) if isinstance(state, dict) else []
    tree = views.get("standalone_polyline_tree")
    if tree is None and not records:
        return
    identities = {record["id"] for record in records}
    if (
        not isinstance(tree, dict)
        or set(tree) != {"items", "current", "expanded", "selected", "current_group"}
        or not isinstance(tree["items"], dict)
        or set(tree["items"]) != identities
        or any(
            not _valid_flag_tuple(flags)
            for flags in tree["items"].values()
        )
        or tree["current"] is not None and (
            not isinstance(tree["current"], str) or tree["current"] not in identities
        )
        or type(tree["expanded"]) is not bool
        or type(tree["selected"]) is not bool
        or type(tree["current_group"]) is not bool
        or tree["current_group"] and tree["current"] is not None
    ):
        raise SessionError("Invalid standalone polyline tree presentation")


def mesh_geometry_values(polydata: vtkPolyData) -> list[np.ndarray]:
    """Borrow source geometry/topology buffers; comparisons stay chunked."""
    from vtkmodules.util.numpy_support import vtk_to_numpy

    values = [vtk_to_numpy(polydata.GetPoints().GetData())]
    for cells in (
        polydata.GetPolys(),
        polydata.GetStrips(),
        polydata.GetLines(),
        polydata.GetVerts(),
    ):
        values.extend(
            [
                vtk_to_numpy(cells.GetOffsetsArray()),
                vtk_to_numpy(cells.GetConnectivityArray()),
            ]
        )
    return values


def mesh_attribute_values(polydata: vtkPolyData) -> list:
    """Include material/UV arrays and active roles without copying numeric payloads."""
    from vtkmodules.util.numpy_support import vtk_to_numpy

    values = []
    for attributes in (polydata.GetPointData(), polydata.GetCellData(),
                       polydata.GetFieldData()):
        arrays = []
        for index in range(attributes.GetNumberOfArrays()):
            array = attributes.GetAbstractArray(index)
            payload = (
                vtk_to_numpy(array) if array.IsA("vtkDataArray")
                else tuple(array.GetVariantValue(i).ToString()
                           for i in range(array.GetNumberOfValues()))
            )
            arrays.append((array.GetName(), array.GetDataType(),
                           array.GetNumberOfComponents(), payload))
        roles = []
        if hasattr(attributes, "GetAttribute"):
            for index in range(attributes.NUM_ATTRIBUTES):
                active = attributes.GetAttribute(index)
                roles.append(active.GetName() if active is not None else None)
        values.append((arrays, roles))
    return values


def restored_mesh_layer(result: MeshLoadResult, record: SessionMeshRecord) -> MeshLayer:
    from .io import SessionError

    layer = create_mesh_layer(
        result, record["coordinate_provenance"]["choice"], layer_id=record["id"],
        parent_id=record.get("parent_id"),
    )
    if layer["coordinate_provenance"] != record["coordinate_provenance"]:
        raise SessionError("Mesh SPACE metadata does not match the saved source")
    for field in ("display_name", "visible", "opacity", "color"):
        layer[field] = record[field]
    if "polylines" in record:
        from copy import deepcopy

        layer["polylines"] = deepcopy(record["polylines"])
    if "presentation" in record:
        layer["presentation"].update(validate_mesh_presentation(record["presentation"]))
    if "texture" in record:
        from copy import deepcopy
        from pathlib import Path

        if Path(result["path"]).suffix.lower() != ".obj":
            raise SessionError("Texture presentation requires an OBJ source")
        layer["texture"] = deepcopy(record["texture"])
    return layer
