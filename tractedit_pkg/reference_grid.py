"""Normalized spatial reference metadata for tractograms and NIfTI outputs."""

from dataclasses import dataclass, field
from typing import Any, Mapping

import nibabel as nib
import numpy as np


def validate_affine(affine: np.ndarray, label: str) -> np.ndarray:
    """Return a finite, invertible affine as float64."""
    matrix = np.asarray(affine, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} affine must be a finite and invertible 4x4 matrix.")
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            f"{label} affine must be a finite and invertible 4x4 matrix."
        ) from error
    if not np.all(np.isfinite(inverse)):
        raise ValueError(f"{label} affine must be a finite and invertible 4x4 matrix.")
    return matrix


def validate_volume_geometry(
    data: Any,
    affine: np.ndarray,
    label: str,
) -> None:
    """Validate the dimensional and spatial contract of a native volume."""
    shape = getattr(data, "shape", None)
    ndim = len(shape) if shape is not None else np.ndim(data)
    if ndim != 3:
        raise ValueError(f"{label} data must have exactly 3 dimensions, got {ndim}.")
    if shape is not None and any(int(length) < 1 for length in shape):
        raise ValueError(f"{label} data dimensions must be non-empty.")

    validate_affine(affine, label)


def canonicalize_nifti(image: nib.Nifti1Image) -> nib.Nifti1Image:
    """Reorder voxels while preserving each declared coordinate system."""
    canonical = nib.as_closest_canonical(image)
    if canonical is image:
        return canonical
    orientation = nib.orientations.ornt_transform(
        nib.io_orientation(image.affine),
        nib.orientations.axcodes2ornt(("R", "A", "S")),
    )
    transform = nib.orientations.inv_ornt_aff(orientation, image.shape)
    sform, scode = image.get_sform(coded=True)
    qform, qcode = image.get_qform(coded=True)
    if scode or qcode:
        canonical.set_sform(
            None if sform is None else sform @ transform,
            int(scode),
            update_affine=False,
        )
        canonical.set_qform(
            None if qform is None else qform @ transform,
            int(qcode),
            strip_shears=False,
            update_affine=False,
        )
    # With neither form declared, retain nibabel's explicit canonical sform:
    # an unset header fallback cannot describe the reordered grid in general.
    return canonical


def _mapping_value(metadata: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in metadata:
            return metadata[key]
    return None


def _shape_from_value(value: Any) -> tuple[int, int, int] | None:
    if isinstance(value, str):
        value = value.translate(str.maketrans("", "", "()[]")).replace(",", " ")
        value = value.split()
    try:
        numeric = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if (
        numeric.size != 3
        or not np.all(np.isfinite(numeric))
        or not np.all(numeric == np.floor(numeric))
        or not np.all(numeric > 0)
    ):
        return None
    return tuple(int(item) for item in numeric)


def _affine_from_value(value: Any) -> np.ndarray | None:
    try:
        affine = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    try:
        validate_affine(affine, "Reference")
    except ValueError:
        return None
    return affine


@dataclass(frozen=True, slots=True, eq=False)
class ReferenceGrid:
    """Authoritative voxel-to-RAS geometry and lightweight provenance."""

    affine: np.ndarray
    shape: tuple[int, int, int]
    provenance: str
    xyzt_units: tuple[str, str] = ("unknown", "unknown")
    _nifti_header: nib.Nifti1Header | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        affine = _affine_from_value(self.affine)
        shape = _shape_from_value(self.shape)
        if affine is None:
            raise ValueError("Reference affine must be a finite 4x4 matrix.")
        if shape is None:
            raise ValueError("Reference shape must contain three positive integers.")
        affine = affine.copy()
        affine.setflags(write=False)
        object.__setattr__(self, "affine", affine)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "xyzt_units", tuple(self.xyzt_units))
        if self._nifti_header is not None:
            object.__setattr__(self, "_nifti_header", self._nifti_header.copy())

    @property
    def voxel_sizes(self) -> tuple[float, float, float]:
        """Return voxel sizes from affine column norms."""
        return tuple(float(value) for value in nib.affines.voxel_sizes(self.affine))

    @property
    def voxel_order(self) -> str:
        """Return the three-letter voxel orientation code."""
        return "".join(nib.aff2axcodes(self.affine)).upper()

    @classmethod
    def from_header(
        cls,
        metadata: Mapping[str, Any] | None,
        provenance: str,
    ) -> "ReferenceGrid | None":
        """Create a grid from normalized TRK or uppercase TRX metadata."""
        if metadata is None:
            return None
        shape = _shape_from_value(_mapping_value(metadata, "dimensions", "DIMENSIONS"))
        affine = _affine_from_value(
            _mapping_value(metadata, "voxel_to_rasmm", "VOXEL_TO_RASMM")
        )
        if shape is None or affine is None:
            return None
        return cls(affine=affine, shape=shape, provenance=provenance)

    @classmethod
    def from_nifti(
        cls,
        image: nib.spatialimages.SpatialImage,
        provenance: str,
    ) -> "ReferenceGrid":
        """Create a grid from an image without materializing voxel data."""
        units = image.header.get_xyzt_units()
        return cls(
            affine=image.affine,
            shape=image.shape[:3],
            provenance=provenance,
            xyzt_units=(units[0] or "unknown", units[1] or "unknown"),
            _nifti_header=image.header,
        )

    @classmethod
    def from_bounds(
        cls,
        minimum: np.ndarray,
        maximum: np.ndarray,
        provenance: str,
        affine: np.ndarray | None = None,
    ) -> "ReferenceGrid":
        """Create a covering grid while preserving a valid nonnegative origin."""
        base_affine = np.eye(4) if affine is None else _affine_from_value(affine)
        if base_affine is None:
            raise ValueError("Synthetic reference affine is invalid.")
        world_bounds = np.asarray([minimum, maximum], dtype=np.float64)
        if world_bounds.shape != (2, 3) or not np.all(np.isfinite(world_bounds)):
            raise ValueError("Synthetic reference bounds must be finite 3D values.")
        corners = (
            np.array(np.meshgrid(*zip(world_bounds[0], world_bounds[1]), indexing="ij"))
            .reshape(3, -1)
            .T
        )
        voxel_bounds = nib.affines.apply_affine(np.linalg.inv(base_affine), corners)
        lower = np.minimum(np.floor(np.min(voxel_bounds, axis=0)), 0).astype(int)
        upper = np.ceil(np.max(voxel_bounds, axis=0)).astype(int)
        shape = tuple(int(value) for value in upper - lower + 1)
        shift = np.eye(4)
        shift[:3, 3] = lower
        return cls(
            affine=base_affine @ shift,
            shape=shape,
            provenance=provenance,
        )

    @classmethod
    def from_points(
        cls,
        points: np.ndarray,
        provenance: str,
        affine: np.ndarray | None = None,
    ) -> "ReferenceGrid":
        """Create a covering grid for world-space points."""
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            raise ValueError("Reference points must be a nonempty Nx3 array.")
        if not np.all(np.isfinite(points)):
            raise ValueError("Reference points must contain finite values.")
        base_affine = np.eye(4) if affine is None else _affine_from_value(affine)
        if base_affine is None:
            raise ValueError("Synthetic reference affine is invalid.")
        if np.array_equal(base_affine, np.eye(4)):
            voxel_points = points
        else:
            voxel_points = nib.affines.apply_affine(
                np.linalg.inv(base_affine),
                points,
            )
        lower = np.minimum(np.floor(np.min(voxel_points, axis=0)), 0).astype(int)
        upper = np.ceil(np.max(voxel_points, axis=0)).astype(int)
        shift = np.eye(4)
        shift[:3, 3] = lower
        return cls(
            affine=base_affine @ shift,
            shape=tuple(int(value) for value in upper - lower + 1),
            provenance=provenance,
        )

    def header_fields(self) -> dict[str, Any]:
        """Return normalized lower-case fields used by TRK/TRX save helpers."""
        return {
            "dimensions": self.shape,
            "voxel_sizes": self.voxel_sizes,
            "voxel_to_rasmm": self.affine.copy(),
            "voxel_order": self.voxel_order,
        }

    def trx_reference(
        self,
        nb_vertices: int,
        nb_streamlines: int,
    ) -> dict[str, Any]:
        """Return the metadata-only reference accepted by trx-python."""
        return {
            "DIMENSIONS": np.asarray(self.shape, dtype=np.uint16),
            "VOXEL_TO_RASMM": self.affine.copy(),
            "NB_VERTICES": int(nb_vertices),
            "NB_STREAMLINES": int(nb_streamlines),
        }

    def create_nifti(self, data: np.ndarray) -> nib.Nifti1Image:
        """Create a NIfTI image whose header is consistent with this grid."""
        if tuple(data.shape[:3]) != self.shape:
            raise ValueError(
                f"Data shape {data.shape[:3]} does not match reference {self.shape}."
            )
        image = nib.Nifti1Image(data, self.affine)
        if self._nifti_header is not None:
            # Copy spatial fields only: anatomical dtype, scaling and intent
            # must not leak into a density image. pixdim encodes the qform's
            # scales, which may differ from the authoritative sform's norms.
            for key in (
                "sform_code",
                "qform_code",
                "srow_x",
                "srow_y",
                "srow_z",
                "quatern_b",
                "quatern_c",
                "quatern_d",
                "qoffset_x",
                "qoffset_y",
                "qoffset_z",
            ):
                image.header[key] = self._nifti_header[key]
            image.header["pixdim"][:4] = self._nifti_header["pixdim"][:4]
        image.header.set_xyzt_units(*self.xyzt_units)
        return image
