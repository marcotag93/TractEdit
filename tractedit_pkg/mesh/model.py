"""Surface-layer identity and scanner-RAS presentation, independent of anatomy."""

import os
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import numpy as np

from ..data_contracts import (
    MeshCoordinateChoice,
    MeshCoordinates,
    MeshLayer,
    MeshLoadResult,
    MeshMetadataSource,
    MeshSpaceMetadata,
)
from .presentation import (
    DEFAULT_MESH_COLOR,
    DEFAULT_MESH_OPACITY,
    default_mesh_presentation,
)

if TYPE_CHECKING:
    from vtkmodules.vtkCommonDataModel import vtkPolyData


MAX_MESH_LAYERS = 256


class MeshCoordinateError(ValueError):
    """A surface cannot be published without an explicit physical-space choice."""


def mesh_space_metadata(
    source: MeshMetadataSource, declarations: Sequence[str] | None
) -> MeshSpaceMetadata:
    """Interpret exact SPACE values; None denotes absence of the declaration.

    Readers supply values from PLY SPACE comments or a VTP string FieldData
    array. VTP requires exactly one value; repeated identical PLY comments agree.
    OBJ has no supported SPACE declaration and always requires an explicit choice.
    """
    if source not in ("ply_comments", "vtp_field_data", "obj_file"):
        raise ValueError("Unsupported mesh SPACE metadata source")
    if isinstance(declarations, str):
        raise TypeError("SPACE declarations must be a sequence of strings")
    values = () if declarations is None else tuple(declarations)
    if any(not isinstance(value, str) for value in values):
        raise TypeError("SPACE declarations must contain only strings")
    metadata: MeshSpaceMetadata = {
        "source": source,
        "declarations": values,
        "status": "missing",
        "coordinate_system": None,
    }
    if source == "obj_file":
        if values:
            raise ValueError("OBJ cannot declare SPACE metadata")
        return metadata
    if declarations is None:
        return metadata
    if not values:
        metadata["status"] = "invalid"
    elif len(set(values)) > 1:
        metadata["status"] = "conflicting"
    elif source == "vtp_field_data" and len(values) != 1:
        metadata["status"] = "invalid"
    elif values[0] not in ("RAS", "LPS"):
        metadata["status"] = "unknown"
    else:
        metadata["status"] = "valid"
        metadata["coordinate_system"] = "RAS" if values[0] == "RAS" else "LPS"
    return metadata


def mesh_metadata_from_vtp(polydata: "vtkPolyData") -> MeshSpaceMetadata:
    """Recognize only the conventional, single-string SPACE FieldData array."""
    array = polydata.GetFieldData().GetAbstractArray("SPACE")
    if array is None:
        return mesh_space_metadata("vtp_field_data", None)
    if not array.IsA("vtkStringArray") or array.GetNumberOfComponents() != 1:
        return mesh_space_metadata("vtp_field_data", ())
    return mesh_space_metadata(
        "vtp_field_data",
        tuple(array.GetValue(index) for index in range(array.GetNumberOfValues())),
    )


def mesh_metadata_from_ply_comments(comments: Sequence[str]) -> MeshSpaceMetadata:
    """Inspect reader-provided comments, never the binary PLY body."""
    values = tuple(comment[6:] for comment in comments if comment.startswith("SPACE="))
    return mesh_space_metadata("ply_comments", values if values else None)


def resolve_mesh_coordinates(
    metadata: MeshSpaceMetadata, choice: MeshCoordinateChoice = "from_file"
) -> MeshCoordinates:
    """Resolve millimetre coordinates without touching source vertices or T1w."""
    if choice not in ("from_file", "RAS", "LPS"):
        raise MeshCoordinateError("Choose From file, RAS (mm) or LPS (mm)")
    if choice == "from_file":
        coordinate_system = metadata["coordinate_system"]
        if metadata["status"] != "valid" or coordinate_system not in ("RAS", "LPS"):
            raise MeshCoordinateError(
                f"File SPACE metadata is {metadata['status']}. "
                "Choose RAS (mm) or LPS (mm) explicitly."
            )
    else:
        coordinate_system = choice
    matrix = np.eye(4, dtype=np.float64)
    if coordinate_system == "LPS":
        matrix[0, 0] = -1.0
        matrix[1, 1] = -1.0
    matrix.flags.writeable = False
    return {
        "coordinate_system": coordinate_system,
        "coordinate_provenance": {
            "choice": choice,
            "metadata": metadata.copy(),
            "overrides_metadata": (
                choice != "from_file"
                and metadata["status"] != "missing"
                and (
                    metadata["status"] != "valid"
                    or metadata["coordinate_system"] != coordinate_system
                )
            ),
        },
        "source_to_ras": matrix,
    }


def create_mesh_layer(
    result: MeshLoadResult,
    choice: MeshCoordinateChoice = "from_file",
    *,
    layer_id: str | None = None,
    layers: Mapping[str, MeshLayer] | None = None,
    parent_id: str | None = None,
) -> MeshLayer:
    """Prepare a layer from validated data; caller owns atomic scene publication."""
    coordinates = resolve_mesh_coordinates(result["space_metadata"], choice)
    if layer_id is None:
        layer_id = str(uuid4())
    else:
        try:
            canonical_id = str(UUID(layer_id))
        except (ValueError, AttributeError, TypeError) as error:
            raise ValueError("Mesh layer ID must be a canonical UUID string") from error
        if canonical_id != layer_id:
            raise ValueError("Mesh layer ID must be a canonical UUID string")
    if layers is not None and parent_id is not None and (
        not isinstance(parent_id, str) or parent_id == layer_id or parent_id not in layers
    ):
        raise ValueError("Invalid mesh parent ID")
    if layers is not None and layer_id in layers:
        raise ValueError("Mesh layer UUID already exists")
    if layers is not None and len(layers) >= MAX_MESH_LAYERS:
        raise ValueError(f"Mesh layer limit reached ({MAX_MESH_LAYERS})")
    path = os.path.abspath(os.fspath(result["path"]))
    basename = os.path.basename(path)
    display_name = basename
    names = {layer["display_name"] for layer in layers.values()} if layers else set()
    suffix = 2
    while display_name in names:
        display_name = f"{basename} ({suffix})"
        suffix += 1
    return {
        **coordinates,
        "id": layer_id,
        "parent_id": parent_id,
        "path": path,
        "display_name": display_name,
        "polydata": result["polydata"],
        "visible": True,
        "opacity": DEFAULT_MESH_OPACITY,
        "color": DEFAULT_MESH_COLOR,
        "presentation": default_mesh_presentation(),
        "point_count": result["point_count"],
        "polygon_count": result["polygon_count"],
        "strip_count": result["strip_count"],
        "source_bounds": result["source_bounds"],
        **({"source_stamp": result["source_stamp"]} if "source_stamp" in result else {}),
        **(
            {"source_sha256": result["source_sha256"]}
            if "source_sha256" in result else {}
        ),
        **(
            {"obj_texture_index": result["obj_texture_index"]}
            if "obj_texture_index" in result else {}
        ),
    }


def _mesh_children(
    layers: Mapping[str, MeshLayer],
) -> dict[str | None, list[str]]:
    if not isinstance(layers, Mapping):
        raise ValueError("Mesh layers must be a mapping")
    children: dict[str | None, list[str]] = {None: []}
    for mesh_id, layer in layers.items():
        try:
            valid_id = isinstance(mesh_id, str) and str(UUID(mesh_id)) == mesh_id
        except ValueError:
            valid_id = False
        if not valid_id or not isinstance(layer, Mapping) or layer.get("id") != mesh_id:
            raise ValueError("Invalid mesh layer identity")
        children[mesh_id] = []
    for mesh_id, layer in layers.items():
        parent_id = layer.get("parent_id")
        if parent_id is not None and (
            not isinstance(parent_id, str) or not parent_id or parent_id not in layers
        ):
            raise ValueError("Mesh parent must reference an existing mesh")
        if parent_id == mesh_id:
            raise ValueError("A mesh cannot be its own parent")
        children[parent_id].append(mesh_id)
    pending = list(children[None])
    visited = 0
    while pending:
        mesh_id = pending.pop()
        visited += 1
        pending.extend(children[mesh_id])
    if visited != len(layers):
        raise ValueError("Mesh hierarchy contains a cycle")
    return children


def validate_mesh_forest(layers: Mapping[str, MeshLayer]) -> None:
    """Validate organizational parent links without inspecting mesh payloads."""
    _mesh_children(layers)


def validate_mesh_parent(
    layers: Mapping[str, MeshLayer], mesh_id: str, parent_id: str | None,
) -> None:
    """Validate a prospective move without changing the forest."""
    validate_mesh_forest(layers)
    if not isinstance(mesh_id, str) or mesh_id not in layers:
        raise ValueError("Mesh does not exist")
    if parent_id is not None and (
        not isinstance(parent_id, str) or not parent_id or parent_id not in layers
    ):
        raise ValueError("Mesh parent must reference an existing mesh")
    ancestor = parent_id
    while ancestor is not None:
        if ancestor == mesh_id:
            raise ValueError("A mesh cannot move under itself or a descendant")
        ancestor = layers[ancestor].get("parent_id")


def mesh_subtree_postorder(
    layers: Mapping[str, MeshLayer], mesh_id: str | None = None,
) -> list[str]:
    """Order a subtree or the entire forest with descendants before parents."""
    children = _mesh_children(layers)
    if mesh_id is not None and (not isinstance(mesh_id, str) or mesh_id not in layers):
        raise ValueError("Mesh does not exist")
    result = []
    roots = children[None] if mesh_id is None else [mesh_id]
    pending = [(root, False) for root in reversed(roots)]
    while pending:
        current, expanded = pending.pop()
        if expanded:
            result.append(current)
        else:
            pending.append((current, True))
            pending.extend((child, False) for child in reversed(children[current]))
    return result


def find_mesh_layer(
    layers: Mapping[str, MeshLayer],
    path: str | os.PathLike[str],
    choice: MeshCoordinateChoice = "from_file",
) -> MeshLayer | None:
    """Find an existing path/convention before a reader is constructed."""
    if choice not in ("from_file", "RAS", "LPS"):
        raise MeshCoordinateError("Choose From file, RAS (mm) or LPS (mm)")
    normalized_path = os.path.normcase(os.path.realpath(path))
    for layer in layers.values():
        if os.path.normcase(os.path.realpath(layer["path"])) != normalized_path:
            continue
        try:
            coordinates = resolve_mesh_coordinates(
                layer["coordinate_provenance"]["metadata"], choice
            )
        except MeshCoordinateError:
            continue
        if coordinates["coordinate_system"] == layer["coordinate_system"]:
            return layer
    return None
