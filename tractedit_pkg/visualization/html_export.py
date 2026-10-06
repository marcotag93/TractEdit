"""Bounded, offline HTML export for TractEdit visualizations.

The exported scene is intended for visual sharing. Streamlines and ROI meshes
may be reduced to keep the single-file artifact safe for a web browser.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from typing import TYPE_CHECKING, Any, Optional

import nibabel as nib
import numpy as np

from ..reference_grid import validate_volume_geometry
from ..transactional_io import staged_output

if TYPE_CHECKING:
    from ..main_window import MainWindow

logger = logging.getLogger(__name__)

DEFAULT_OPTIONS = {
    "max_streamlines": 1000,
    "streamline_step": 2,
    "max_streamline_points": 256,
    "image_quality": 85,
    "max_slice_size": 512,
    "include_slices": True,
    "include_rois": True,
    "max_rois": 64,
    "max_roi_triangles": 20_000,
    "max_total_roi_triangles": 50_000,
    "max_roi_voxels": 4_000_000,
}


def export_to_html(
    main_window: "MainWindow",
    output_path: str,
    options: Optional[dict[str, Any]] = None,
) -> bool:
    """Export the current bounded visualization to one offline HTML file."""
    opts = {**DEFAULT_OPTIONS, **(options or {})}

    try:
        _validate_options(opts)
        data = _collect_visualization_data(main_window, opts)
        if not any((data["streamlines"], data["slices"], data["rois"])):
            logger.warning("No visible data to export.")
            return False

        with staged_output(output_path) as staged_path:
            with open(staged_path, "w", encoding="utf-8") as output_file:
                output_file.write(_generate_html(data, opts))
    except (OSError, ValueError, KeyError, AttributeError, RuntimeError) as error:
        logger.error("HTML export failed: %s", error, exc_info=True)
        return False

    logger.info("HTML export complete: %s", output_path)
    return True


def _validate_options(options: dict[str, Any]) -> None:
    for name in (
        "max_streamlines",
        "streamline_step",
        "max_streamline_points",
        "max_slice_size",
        "max_rois",
        "max_roi_triangles",
        "max_total_roi_triangles",
        "max_roi_voxels",
    ):
        value = options.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"HTML export option {name!r} must be a positive integer.")
    if options["max_streamline_points"] < 2:
        raise ValueError("HTML export requires at least two points per streamline.")
    if options["max_roi_voxels"] < 27:
        raise ValueError("HTML export requires a ROI voxel budget of at least 27.")


def _collect_visualization_data(
    main_window: "MainWindow", options: dict[str, Any]
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "streamlines": [],
        "streamline_colors": [],
        "slices": {},
        "slice_positions": {},
        "rois": [],
        "metadata": {},
    }

    visible_indices = getattr(main_window, "visible_indices", set())
    if (
        getattr(main_window, "tractogram_data", None) is not None
        and visible_indices
        and getattr(main_window, "bundle_is_visible", True)
    ):
        streamlines, colors = _subsample_streamlines(main_window, options)
        data["streamlines"] = streamlines
        data["streamline_colors"] = colors
        data["metadata"]["num_streamlines"] = len(streamlines)
        data["metadata"]["total_streamlines"] = len(visible_indices)

    if (
        options.get("include_slices", True)
        and getattr(main_window, "vtk_panel", None)
        and getattr(main_window, "image_is_visible", True)
    ):
        slices, positions = _capture_slice_images(main_window, options)
        data["slices"] = slices
        data["slice_positions"] = positions

    roi_layers = getattr(main_window, "roi_layers", {})
    if options.get("include_rois", True) and roi_layers:
        data["rois"] = _collect_roi_data(main_window, options)
        data["metadata"]["num_rois"] = len(data["rois"])

    return data


def _subsample_streamlines(
    main_window: "MainWindow", options: dict[str, Any]
) -> tuple[list[list[list[float]]], list[list[int]]]:
    """Return a deterministic, endpoint-preserving bounded subset."""
    max_streamlines = options.get("max_streamlines", 1000)
    point_step = options.get("streamline_step", 2)
    max_points = options.get("max_streamline_points", 256)
    visible_indices = sorted(main_window.visible_indices)

    if len(visible_indices) > max_streamlines:
        positions = np.linspace(
            0, len(visible_indices) - 1, max_streamlines, dtype=np.int64
        )
        selected_indices = [visible_indices[position] for position in positions]
    else:
        selected_indices = visible_indices

    streamlines: list[list[list[float]]] = []
    colors: list[list[int]] = []
    for index in selected_indices:
        original = np.asarray(main_window.tractogram_data[index])
        if original.ndim != 2 or original.shape[1] != 3 or len(original) < 2:
            continue

        point_indices = np.arange(0, len(original), point_step, dtype=np.int64)
        if point_indices[-1] != len(original) - 1:
            point_indices = np.append(point_indices, len(original) - 1)
        if len(point_indices) > max_points:
            keep = np.linspace(0, len(point_indices) - 1, max_points, dtype=np.int64)
            point_indices = point_indices[keep]
        streamline = original[point_indices]
        streamlines.append(streamline.tolist())

        direction = np.abs(original[-1] - original[0]).astype(np.float64)
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction /= norm
            colors.append(np.rint(direction * 255).astype(int).tolist())
        else:
            colors.append([200, 200, 200])

    return streamlines, colors


def _slice_plane_corners(
    affine: np.ndarray,
    shape: tuple[int, int, int],
    axis: int,
    index: int,
) -> list[list[float]]:
    lower = np.full(3, -0.5, dtype=np.float64)
    upper = np.asarray(shape, dtype=np.float64) - 0.5
    if axis == 2:
        voxels = np.array(
            [
                [lower[0], lower[1], index],
                [upper[0], lower[1], index],
                [upper[0], upper[1], index],
                [lower[0], upper[1], index],
            ]
        )
    elif axis == 1:
        voxels = np.array(
            [
                [lower[0], index, lower[2]],
                [upper[0], index, lower[2]],
                [upper[0], index, upper[2]],
                [lower[0], index, upper[2]],
            ]
        )
    else:
        voxels = np.array(
            [
                [index, lower[1], lower[2]],
                [index, upper[1], lower[2]],
                [index, upper[1], upper[2]],
                [index, lower[1], upper[2]],
            ]
        )
    return nib.affines.apply_affine(affine, voxels).tolist()


def _slice_to_data_url(
    slice_data: np.ndarray,
    image_quality: int,
    max_size: int,
) -> str:
    from PIL import Image

    array = np.asarray(slice_data, dtype=np.float32)
    row_step = max(1, int(np.ceil(array.shape[0] / max_size)))
    column_step = max(1, int(np.ceil(array.shape[1] / max_size)))
    array = array[::row_step, ::column_step]
    finite = np.isfinite(array)
    if finite.any():
        values = array[finite]
        lower, upper = np.percentile(values, (1.0, 99.0))
        if upper <= lower:
            lower = float(values.min())
            upper = float(values.max())
        if upper > lower:
            scaled = np.clip((array - lower) / (upper - lower), 0.0, 1.0)
        else:
            scaled = np.zeros_like(array)
    else:
        scaled = np.zeros_like(array)
    scaled[~finite] = 0.0
    pixels = np.rint(scaled * 255).astype(np.uint8)

    buffer = io.BytesIO()
    Image.fromarray(pixels).save(
        buffer,
        format="JPEG",
        quality=int(np.clip(image_quality, 1, 100)),
    )
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _capture_slice_images(
    main_window: "MainWindow", options: dict[str, Any]
) -> tuple[dict[str, str], dict[str, Any]]:
    """Encode current anatomical planes without mutating or rendering the GUI."""
    image_data = getattr(main_window, "anatomical_image_data", None)
    affine = getattr(main_window, "anatomical_image_affine", None)
    if image_data is None or affine is None:
        return {}, {}
    validate_volume_geometry(image_data, affine, "Anatomical image")

    shape = tuple(int(value) for value in image_data.shape)
    current = getattr(main_window.vtk_panel, "current_slice_indices", {})
    indices = {
        "sagittal": int(current.get("x", shape[0] // 2)),
        "coronal": int(current.get("y", shape[1] // 2)),
        "axial": int(current.get("z", shape[2] // 2)),
    }
    indices["sagittal"] = int(np.clip(indices["sagittal"], 0, shape[0] - 1))
    indices["coronal"] = int(np.clip(indices["coronal"], 0, shape[1] - 1))
    indices["axial"] = int(np.clip(indices["axial"], 0, shape[2] - 1))

    planes = {
        "axial": np.asarray(image_data[:, :, indices["axial"]]).T,
        "coronal": np.asarray(image_data[:, indices["coronal"], :]).T,
        "sagittal": np.asarray(image_data[indices["sagittal"], :, :]).T,
    }
    axes = {"axial": 2, "coronal": 1, "sagittal": 0}
    slices = {
        name: _slice_to_data_url(
            plane,
            options.get("image_quality", 85),
            options.get("max_slice_size", 512),
        )
        for name, plane in planes.items()
    }
    positions = {
        name: {
            "corners": _slice_plane_corners(
                np.asarray(affine, dtype=np.float64),
                shape,
                axes[name],
                indices[name],
            ),
            "index": indices[name],
        }
        for name in planes
    }
    return slices, positions


def _foreground_bounds(
    data: np.ndarray,
    block_bytes: int = 4 * 1024 * 1024,
) -> tuple[np.ndarray, np.ndarray] | None:
    lower = np.asarray(data.shape, dtype=np.int64)
    upper = np.full(3, -1, dtype=np.int64)
    plane_voxels = max(1, int(data.shape[1]) * int(data.shape[2]))
    block_width = max(1, block_bytes // plane_voxels)
    for start in range(0, data.shape[0], block_width):
        stop = min(start + block_width, data.shape[0])
        foreground = np.asarray(data[start:stop]) > 0
        occupied = (
            np.flatnonzero(np.any(foreground, axis=(1, 2))),
            np.flatnonzero(np.any(foreground, axis=(0, 2))),
            np.flatnonzero(np.any(foreground, axis=(0, 1))),
        )
        if occupied[0].size:
            lower[0] = min(lower[0], start + occupied[0][0])
            upper[0] = max(upper[0], start + occupied[0][-1])
        for axis in (1, 2):
            if occupied[axis].size:
                lower[axis] = min(lower[axis], occupied[axis][0])
                upper[axis] = max(upper[axis], occupied[axis][-1])
    if np.any(upper < lower):
        return None
    return lower, upper + 1


def _bounded_roi_mask(
    data: np.ndarray,
    max_voxels: int,
) -> tuple[np.ndarray, np.ndarray, int] | None:
    if max_voxels < 27:
        raise ValueError("A padded ROI surface requires at least 27 voxels.")
    bounds = _foreground_bounds(data)
    if bounds is None:
        return None
    lower, upper = bounds
    shape = upper - lower
    step = 1
    while np.prod((shape + step - 1) // step + 2) > max_voxels:
        step += 1
    coarse_shape = (shape + step - 1) // step
    mask = np.zeros(tuple(coarse_shape + 2), dtype=np.uint8)

    # Source tiles bound temporary predicates and index arrays. Every positive
    # voxel marks its containing coarse cell, regardless of stride residue.
    tile_budget = 100_000
    tile_z = min(int(shape[2]), tile_budget)
    tile_y = min(int(shape[1]), max(1, tile_budget // tile_z))
    tile_x = min(int(shape[0]), max(1, tile_budget // (tile_y * tile_z)))
    for x in range(int(lower[0]), int(upper[0]), tile_x):
        for y in range(int(lower[1]), int(upper[1]), tile_y):
            for z in range(int(lower[2]), int(upper[2]), tile_z):
                source = data[
                    x : min(x + tile_x, int(upper[0])),
                    y : min(y + tile_y, int(upper[1])),
                    z : min(z + tile_z, int(upper[2])),
                ]
                occupied = np.nonzero(np.asarray(source) > 0)
                if occupied[0].size:
                    mask[
                        (occupied[0] + x - lower[0]) // step + 1,
                        (occupied[1] + y - lower[1]) // step + 1,
                        (occupied[2] + z - lower[2]) // step + 1,
                    ] = 1
    voxel_to_source = np.eye(4)
    voxel_to_source[:3, :3] *= step
    voxel_to_source[:3, 3] = lower - (step + 1) / 2
    return mask, voxel_to_source, step


def _polydata_triangles(polydata: Any) -> tuple[np.ndarray, np.ndarray]:
    from vtk.util.numpy_support import vtk_to_numpy

    if polydata.GetNumberOfPoints() == 0 or polydata.GetNumberOfPolys() == 0:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.int64)
    points = vtk_to_numpy(polydata.GetPoints().GetData())
    polygons = polydata.GetPolys()
    connectivity = vtk_to_numpy(polygons.GetConnectivityArray())
    offsets = vtk_to_numpy(polygons.GetOffsetsArray())
    lengths = np.diff(offsets)
    if len(lengths) and not np.all(lengths == 3):
        raise ValueError("ROI surface contains non-triangular cells.")
    return points, connectivity.reshape(-1, 3)


def _build_roi_mesh(
    roi_data: np.ndarray,
    affine: np.ndarray,
    max_triangles: int,
    max_voxels: int,
) -> dict[str, Any] | None:
    """Build a bounded surface mesh from the physical ROI voxel support."""
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    crop = _bounded_roi_mask(np.asarray(roi_data), max_voxels)
    if crop is None:
        return None
    mask, voxel_to_source, voxel_stride = crop

    image = vtk.vtkImageData()
    image.SetDimensions(*mask.shape)
    scalars = numpy_to_vtk(
        np.ascontiguousarray(mask, dtype=np.uint8).ravel(order="F"),
        deep=True,
    )
    image.GetPointData().SetScalars(scalars)

    surface = vtk.vtkFlyingEdges3D()
    surface.SetInputData(image)
    surface.SetValue(0, 0.5)
    surface.ComputeNormalsOff()
    surface.Update()

    triangulate = vtk.vtkTriangleFilter()
    triangulate.SetInputConnection(surface.GetOutputPort())
    triangulate.Update()
    polydata = triangulate.GetOutput()
    initial_triangles = polydata.GetNumberOfPolys()
    if initial_triangles == 0 or polydata.GetNumberOfPoints() == 0:
        raise ValueError("Occupied ROI produced no surface for HTML export.")
    decimated = voxel_stride > 1 or initial_triangles > max_triangles
    if initial_triangles > max_triangles:
        reduction = 1.0 - (max_triangles / initial_triangles)
        simplify = vtk.vtkQuadricDecimation()
        simplify.SetInputData(polydata)
        simplify.SetTargetReduction(float(np.clip(reduction, 0.0, 0.999)))
        simplify.Update()
        if simplify.GetOutput().GetNumberOfPolys() > 0:
            polydata = simplify.GetOutput()

    points, triangles = _polydata_triangles(polydata)
    if len(triangles) == 0:
        raise ValueError("Occupied ROI surface disappeared during simplification.")
    if len(triangles) > max_triangles:
        keep = np.linspace(0, len(triangles) - 1, max_triangles, dtype=np.int64)
        triangles = triangles[keep]
        decimated = True

    used_points = np.unique(triangles)
    remap = np.full(len(points), -1, dtype=np.int64)
    remap[used_points] = np.arange(len(used_points), dtype=np.int64)
    points = points[used_points]
    triangles = remap[triangles]

    world_points = nib.affines.apply_affine(affine @ voxel_to_source, points)
    return {
        "points": world_points.tolist(),
        "triangles": triangles.astype(np.int64, copy=False).tolist(),
        "decimated": decimated,
        "voxel_stride": voxel_stride,
        "occupied_cell_aggregation": voxel_stride > 1,
    }


def _roi_color(layer: dict[str, Any]) -> list[int]:
    color = np.asarray(layer.get("color", (1.0, 0.0, 0.0)), dtype=np.float64)
    if color.shape[0] < 3 or not np.all(np.isfinite(color[:3])):
        raise ValueError("ROI color must contain three finite values.")
    return np.rint(np.clip(color[:3], 0.0, 1.0) * 255).astype(int).tolist()


def _collect_roi_data(
    main_window: "MainWindow", options: dict[str, Any]
) -> list[dict[str, Any]]:
    visible_names = [
        name
        for name in main_window.roi_layers
        if getattr(main_window, "roi_visibility", {}).get(name, True)
    ]
    if len(visible_names) > options.get("max_rois", 64):
        raise ValueError(
            f"HTML export has {len(visible_names)} visible ROIs; "
            f"the configured limit is {options.get('max_rois', 64)}."
        )

    panel = getattr(main_window, "vtk_panel", None)
    sphere_params = getattr(panel, "sphere_params_per_roi", {})
    rectangle_params = getattr(panel, "rectangle_params_per_roi", {})
    remaining_triangles = options.get("max_total_roi_triangles", 50_000)
    per_roi_limit = options.get("max_roi_triangles", 20_000)
    max_roi_voxels = options.get("max_roi_voxels", 4_000_000)
    rois: list[dict[str, Any]] = []

    for name in visible_names:
        layer = main_window.roi_layers[name]
        data = np.asarray(layer["data"])
        affine = np.asarray(layer["affine"], dtype=np.float64)
        validate_volume_geometry(data, affine, f"ROI {name!r}")
        color = _roi_color(layer)
        if name in sphere_params:
            params = sphere_params[name]
            center = np.asarray(params["center"], dtype=np.float64)
            radius = float(params["radius"])
            if center.shape != (3,) or not np.all(np.isfinite(center)):
                raise ValueError(f"ROI {name!r} has an invalid sphere center.")
            if not np.isfinite(radius) or radius <= 0:
                raise ValueError(f"ROI {name!r} has an invalid sphere radius.")
            rois.append(
                {
                    "name": name,
                    "type": "sphere",
                    "center": center.tolist(),
                    "radius": radius,
                    "color": color,
                }
            )
            continue

        if (
            name in rectangle_params
            and rectangle_params[name].get("corners") is not None
        ):
            params = rectangle_params[name]
            corners = np.asarray(params["corners"], dtype=np.float64)
            if corners.shape != (4, 3) or not np.all(np.isfinite(corners)):
                raise ValueError(f"ROI {name!r} has invalid rectangle corners.")
            rois.append(
                {
                    "name": name,
                    "type": "polygon",
                    "corners": corners.tolist(),
                    "color": color,
                }
            )
            continue

        if remaining_triangles < 1:
            raise ValueError("Visible ROI surfaces exceed the HTML triangle budget.")
        mesh = _build_roi_mesh(
            data,
            affine,
            min(per_roi_limit, remaining_triangles),
            max_roi_voxels,
        )
        if mesh is None:
            continue
        remaining_triangles -= len(mesh["triangles"])
        rois.append(
            {
                "name": name,
                "type": "mesh",
                "color": color,
                **mesh,
            }
        )

    return rois


def _json_for_script(value: Any) -> str:
    return (
        json.dumps(value, separators=(",", ":"), allow_nan=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def _generate_html(data: dict[str, Any], options: dict[str, Any]) -> str:
    """Return a dependency-free WebGL document containing the scene payload."""
    del options
    return _HTML_TEMPLATE.replace("__SCENE_DATA__", _json_for_script(data))


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TractEdit offline visual export</title>
<style>
*{box-sizing:border-box}html,body{height:100%;margin:0;background:#1a1a2e;color:#eee;font:14px system-ui,sans-serif}#layout{height:100%;display:flex}#viewer{min-width:0;flex:1;position:relative}canvas{display:block;width:100%;height:100%}#side{width:290px;padding:18px;background:#16213e;overflow:auto}h1{font-size:18px;color:#e94560;margin:0 0 18px}.section{border-bottom:1px solid #315078;padding:0 0 14px;margin:0 0 14px}label{display:block;margin:8px 0}.note{color:#b5bfd0;font-size:12px;line-height:1.45}input{margin-right:8px}#error{position:absolute;inset:20px auto auto 20px;color:#ff8b8b}</style>
</head>
<body><div id="layout"><div id="viewer"><canvas id="canvas"></canvas><div id="error"></div></div><aside id="side">
<h1>TractEdit</h1><div class="section"><label><input id="streamlines" type="checkbox" checked>Streamlines</label><label><input id="rois" type="checkbox" checked>ROIs</label><label><input id="slices" type="checkbox" checked>Anatomical slices</label></div>
<div class="section"><label>Streamline opacity <input id="opacity" type="range" min="0" max="100" value="80"></label></div>
<div class="section note">Left drag: rotate<br>Right drag: pan<br>Wheel: zoom</div>
<div class="note"><strong>Offline visual export.</strong> No external JavaScript libraries or network access are required. This is not a quantitative scientific export: streamlines and textures may be downsampled; occupied ROI voxels are grouped into bounded cells and surfaces may be decimated. Coarse ROI cell boundaries can extend beyond source voxels. Source coordinates and affine-transformed plane geometry are retained.</div>
</aside></div>
<script>
'use strict';
const sceneData=__SCENE_DATA__;
const canvas=document.getElementById('canvas');
const gl=canvas.getContext('webgl',{alpha:false,antialias:true});
if(!gl){document.getElementById('error').textContent='WebGL is unavailable in this browser.';throw new Error('WebGL unavailable');}
const vertexSource='attribute vec3 p;attribute vec2 uv;uniform mat4 vp;varying vec2 t;void main(){t=uv;gl_Position=vp*vec4(p,1.0);}';
const fragmentSource='precision mediump float;varying vec2 t;uniform vec4 color;uniform sampler2D image;uniform bool textured;void main(){gl_FragColor=textured?texture2D(image,t)*color:color;}';
function shader(type,source){const value=gl.createShader(type);gl.shaderSource(value,source);gl.compileShader(value);if(!gl.getShaderParameter(value,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(value));return value;}
const program=gl.createProgram();gl.attachShader(program,shader(gl.VERTEX_SHADER,vertexSource));gl.attachShader(program,shader(gl.FRAGMENT_SHADER,fragmentSource));gl.linkProgram(program);gl.useProgram(program);
const loc={p:gl.getAttribLocation(program,'p'),uv:gl.getAttribLocation(program,'uv'),vp:gl.getUniformLocation(program,'vp'),color:gl.getUniformLocation(program,'color'),image:gl.getUniformLocation(program,'image'),textured:gl.getUniformLocation(program,'textured')};
function buffer(values,size){const result=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,result);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(values),gl.STATIC_DRAW);return {value:result,size:size,count:values.length/size};}
function normalize(v){const n=Math.hypot(...v)||1;return v.map(x=>x/n);}
function cross(a,b){return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];}
function dot(a,b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
function multiply(a,b){const out=new Array(16).fill(0);for(let c=0;c<4;c++)for(let r=0;r<4;r++)for(let k=0;k<4;k++)out[c*4+r]+=a[k*4+r]*b[c*4+k];return out;}
function perspective(fovy,aspect,near,far){const f=1/Math.tan(fovy/2),nf=1/(near-far);return [f/aspect,0,0,0,0,f,0,0,0,0,(far+near)*nf,-1,0,0,2*far*near*nf,0];}
function lookAt(eye,target,up){const z=normalize(eye.map((v,i)=>v-target[i])),x=normalize(cross(up,z)),y=cross(z,x);return [x[0],y[0],z[0],0,x[1],y[1],z[1],0,x[2],y[2],z[2],0,-dot(x,eye),-dot(y,eye),-dot(z,eye),1];}
const objects=[];const bounds=[Infinity,Infinity,Infinity,-Infinity,-Infinity,-Infinity];
function include(point){for(let i=0;i<3;i++){bounds[i]=Math.min(bounds[i],point[i]);bounds[i+3]=Math.max(bounds[i+3],point[i]);}}
function add(vertices,mode,color,group,texcoords=null,texture=null){vertices.forEach(include);objects.push({positions:buffer(vertices.flat(),3),uv:buffer((texcoords||vertices.map(()=>[0,0])).flat(),2),mode:mode,color:color.map(x=>x/255),group:group,texture:texture});}
sceneData.streamlines.forEach((line,index)=>{const segments=[];for(let i=1;i<line.length;i++)segments.push(line[i-1],line[i]);add(segments,gl.LINES,[...(sceneData.streamline_colors[index]||[200,200,200]),204],'streamlines');});
function sphere(center,radius){const points=[],triangles=[],rows=12,cols=20;for(let r=0;r<=rows;r++){const phi=Math.PI*r/rows;for(let c=0;c<=cols;c++){const theta=2*Math.PI*c/cols;points.push([center[0]+radius*Math.sin(phi)*Math.cos(theta),center[1]+radius*Math.sin(phi)*Math.sin(theta),center[2]+radius*Math.cos(phi)]);}}for(let r=0;r<rows;r++)for(let c=0;c<cols;c++){const a=r*(cols+1)+c,b=a+cols+1;triangles.push(points[a],points[b],points[a+1],points[a+1],points[b],points[b+1]);}return triangles;}
sceneData.rois.forEach(roi=>{let vertices=[];if(roi.type==='sphere')vertices=sphere(roi.center,roi.radius);else if(roi.type==='polygon')vertices=[roi.corners[0],roi.corners[1],roi.corners[2],roi.corners[0],roi.corners[2],roi.corners[3]];else roi.triangles.forEach(t=>vertices.push(roi.points[t[0]],roi.points[t[1]],roi.points[t[2]]));add(vertices,gl.TRIANGLES,[...roi.color,128],'rois');});
Object.entries(sceneData.slices).forEach(([name,source])=>{const info=sceneData.slice_positions[name];if(!info)return;const c=info.corners,vertices=[c[0],c[1],c[2],c[0],c[2],c[3]],uv=[[0,0],[1,0],[1,1],[0,0],[1,1],[0,1]],texture=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,texture);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,1,1,0,gl.RGBA,gl.UNSIGNED_BYTE,new Uint8Array([80,80,80,255]));const image=new Image();image.onload=()=>{gl.bindTexture(gl.TEXTURE_2D,texture);gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,true);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,image);draw();};image.src=source;add(vertices,gl.TRIANGLES,[255,255,255,190],'slices',uv,texture);});
if(!Number.isFinite(bounds[0]))bounds.splice(0,6,-1,-1,-1,1,1,1);let target=[(bounds[0]+bounds[3])/2,(bounds[1]+bounds[4])/2,(bounds[2]+bounds[5])/2],distance=Math.max(bounds[3]-bounds[0],bounds[4]-bounds[1],bounds[5]-bounds[2],1)*1.8,yaw=-Math.PI/2,pitch=.3;
gl.enable(gl.DEPTH_TEST);gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);gl.clearColor(.102,.102,.18,1);
function draw(){const ratio=Math.min(devicePixelRatio||1,2),width=Math.max(1,Math.floor(canvas.clientWidth*ratio)),height=Math.max(1,Math.floor(canvas.clientHeight*ratio));if(canvas.width!==width||canvas.height!==height){canvas.width=width;canvas.height=height;}gl.viewport(0,0,width,height);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);const cp=Math.cos(pitch),eye=[target[0]+distance*cp*Math.cos(yaw),target[1]+distance*cp*Math.sin(yaw),target[2]+distance*Math.sin(pitch)],vp=multiply(perspective(Math.PI/3,width/height,.01,Math.max(10000,distance*20)),lookAt(eye,target,[0,0,1]));gl.uniformMatrix4fv(loc.vp,false,new Float32Array(vp));for(const item of objects){if(!document.getElementById(item.group).checked)continue;gl.bindBuffer(gl.ARRAY_BUFFER,item.positions.value);gl.enableVertexAttribArray(loc.p);gl.vertexAttribPointer(loc.p,3,gl.FLOAT,false,0,0);gl.bindBuffer(gl.ARRAY_BUFFER,item.uv.value);gl.enableVertexAttribArray(loc.uv);gl.vertexAttribPointer(loc.uv,2,gl.FLOAT,false,0,0);const alpha=item.group==='streamlines'?Number(document.getElementById('opacity').value)/100:item.color[3];gl.uniform4f(loc.color,item.color[0],item.color[1],item.color[2],alpha);gl.uniform1i(loc.textured,item.texture?1:0);if(item.texture){gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,item.texture);gl.uniform1i(loc.image,0);}gl.drawArrays(item.mode,0,item.positions.count);}}
let drag=null;canvas.addEventListener('contextmenu',event=>event.preventDefault());canvas.addEventListener('pointerdown',event=>{drag={x:event.clientX,y:event.clientY,button:event.button};canvas.setPointerCapture(event.pointerId);});canvas.addEventListener('pointerup',()=>drag=null);canvas.addEventListener('pointermove',event=>{if(!drag)return;const dx=event.clientX-drag.x,dy=event.clientY-drag.y;drag.x=event.clientX;drag.y=event.clientY;if(drag.button===0){yaw-=dx*.008;pitch=Math.max(-1.5,Math.min(1.5,pitch+dy*.008));}else{const scale=distance*.0015,right=[-Math.sin(yaw),Math.cos(yaw),0],up=normalize(cross(right,[Math.cos(pitch)*Math.cos(yaw),Math.cos(pitch)*Math.sin(yaw),Math.sin(pitch)]));for(let i=0;i<3;i++)target[i]+=(-dx*right[i]+dy*up[i])*scale;}draw();});canvas.addEventListener('wheel',event=>{event.preventDefault();distance*=Math.exp(event.deltaY*.001);distance=Math.max(distance,.01);draw();},{passive:false});['streamlines','rois','slices','opacity'].forEach(id=>document.getElementById(id).addEventListener('input',draw));window.addEventListener('resize',draw);draw();
</script></body></html>"""
