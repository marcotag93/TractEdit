# -*- coding: utf-8 -*-

"""
HTML Export Module for TractEdit.

This is **EXPERIMENTAL** and may not work as expected. Currently it can handle the bundle and the 2D slices.
In the future updates it will be able to use a less heavy resampling and to handle the 3D slices and the ROIs.

Exports the current visualization (streamlines, anatomical slices, ROIs)
to a self-contained interactive HTML file using three.js for WebGL rendering.

Data is subsampled to keep file sizes reasonable for web viewing.
"""

import os
import io
import json
import base64
import logging
import gzip
from typing import TYPE_CHECKING, Optional, List, Dict, Any, Tuple
import numpy as np

if TYPE_CHECKING:
    from ..main_window import MainWindow

logger = logging.getLogger(__name__)

# Default export options
DEFAULT_OPTIONS = {
    "max_streamlines": 1000,  # Maximum streamlines to export
    "streamline_step": 2,  # Keep every Nth point along streamline
    "image_quality": 85,  # JPEG quality for slices
    "include_slices": True,  # Include 2D slice images
    "include_rois": True,  # Include ROI visualizations
}


def export_to_html(
    main_window: "MainWindow",
    output_path: str,
    options: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Exports the current visualization to an interactive HTML file.

    Args:
        main_window: Reference to the MainWindow instance.
        output_path: Path for the output HTML file.
        options: Optional dictionary of export options.

    Returns:
        True if export was successful, False otherwise.
    """
    opts = {**DEFAULT_OPTIONS, **(options or {})}

    try:
        logger.info(f"Starting HTML export to: {output_path}")

        # Collect data
        data = _collect_visualization_data(main_window, opts)

        if not data["streamlines"] and not data["slices"]:
            logger.warning("No data to export.")
            return False

        # Generate HTML
        html_content = _generate_html(data, opts)

        # Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"HTML export complete: {output_path}")
        return True

    except Exception as e:
        logger.error(f"HTML export failed: {e}", exc_info=True)
        return False


def _collect_visualization_data(
    main_window: "MainWindow", options: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Collects and subsamples visualization data for export.

    Returns:
        Dictionary containing streamlines, slices, and ROI data.
    """
    data = {
        "streamlines": [],
        "streamline_colors": [],
        "slices": {},
        "slice_positions": {},
        "rois": [],
        "metadata": {},
    }

    # Collect streamlines
    if main_window.tractogram_data and main_window.visible_indices:
        data["streamlines"], data["streamline_colors"] = _subsample_streamlines(
            main_window, options
        )
        data["metadata"]["num_streamlines"] = len(data["streamlines"])
        data["metadata"]["total_streamlines"] = len(main_window.visible_indices)

    # Collect slice images and positions
    if options.get("include_slices", True) and main_window.vtk_panel:
        data["slices"], data["slice_positions"] = _capture_slice_images(
            main_window, options
        )

    # Collect ROI data
    if options.get("include_rois", True) and main_window.roi_layers:
        data["rois"] = _collect_roi_data(main_window, options)

    return data


def _subsample_streamlines(
    main_window: "MainWindow", options: Dict[str, Any]
) -> Tuple[List[List[List[float]]], List[List[int]]]:
    """
    Subsamples streamlines for web export.

    Returns:
        Tuple of (streamlines, colors) where each streamline is a list of [x,y,z] points.
    """
    max_sl = options.get("max_streamlines", 1000)
    step = options.get("streamline_step", 2)

    visible_indices = list(main_window.visible_indices)
    tractogram = main_window.tractogram_data

    # Subsample indices
    if len(visible_indices) > max_sl:
        # Uniform sampling
        indices = np.linspace(0, len(visible_indices) - 1, max_sl, dtype=int)
        selected_indices = [visible_indices[i] for i in indices]
    else:
        selected_indices = visible_indices

    streamlines = []
    colors = []

    for idx in selected_indices:
        sl = tractogram[idx]
        if sl is None or len(sl) < 2:
            continue

        # Subsample points along streamline
        if step > 1:
            sl = sl[::step]

        # Convert to list for JSON serialization
        streamlines.append(sl.tolist())

        # Generate color (direction-based RGB)
        if len(sl) >= 2:
            direction = sl[-1] - sl[0]
            direction = np.abs(direction)
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
            color = [int(c * 255) for c in direction]
        else:
            color = [200, 200, 200]
        colors.append(color)

    return streamlines, colors


def _capture_slice_images(
    main_window: "MainWindow", options: Dict[str, Any]
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """
    Captures current slice images as base64 encoded PNGs.

    Returns:
        Tuple of (slices dict, slice_positions dict).
    """
    slices = {}
    slice_positions = {}

    try:
        from PIL import Image

        vtk_panel = main_window.vtk_panel
        if not vtk_panel:
            return slices, slice_positions

        quality = options.get("image_quality", 85)

        # Get current slice positions and image bounds
        if (
            main_window.anatomical_image_data is not None
            and main_window.anatomical_image_affine is not None
        ):
            affine = main_window.anatomical_image_affine
            shape = main_window.anatomical_image_data.shape[:3]
            current_indices = vtk_panel.current_slice_indices

            # Compute world coordinates for each slice
            x_idx = current_indices.get("x", shape[0] // 2)
            y_idx = current_indices.get("y", shape[1] // 2)
            z_idx = current_indices.get("z", shape[2] // 2)

            # Compute world bounds by transforming all 8 corners
            corners_vox = np.array(
                [
                    [0, 0, 0, 1],
                    [shape[0] - 1, 0, 0, 1],
                    [0, shape[1] - 1, 0, 1],
                    [0, 0, shape[2] - 1, 1],
                    [shape[0] - 1, shape[1] - 1, 0, 1],
                    [shape[0] - 1, 0, shape[2] - 1, 1],
                    [0, shape[1] - 1, shape[2] - 1, 1],
                    [shape[0] - 1, shape[1] - 1, shape[2] - 1, 1],
                ]
            )
            corners_world = np.array([np.dot(affine, c)[:3] for c in corners_vox])

            # Get actual min/max across all corners
            world_min = corners_world.min(axis=0)
            world_max = corners_world.max(axis=0)

            # Axial slice position (fixed Z)
            z_vox = np.array([shape[0] / 2, shape[1] / 2, z_idx, 1])
            z_world = np.dot(affine, z_vox)[:3]
            slice_positions["axial"] = {
                "position": float(z_world[2]),
                "axis": "z",
                "bounds": {
                    "minX": float(world_min[0]),
                    "maxX": float(world_max[0]),
                    "minY": float(world_min[1]),
                    "maxY": float(world_max[1]),
                },
            }

            # Coronal slice position (fixed Y)
            y_vox = np.array([shape[0] / 2, y_idx, shape[2] / 2, 1])
            y_world = np.dot(affine, y_vox)[:3]
            slice_positions["coronal"] = {
                "position": float(y_world[1]),
                "axis": "y",
                "bounds": {
                    "minX": float(world_min[0]),
                    "maxX": float(world_max[0]),
                    "minZ": float(world_min[2]),
                    "maxZ": float(world_max[2]),
                },
            }

            # Sagittal slice position (fixed X)
            x_vox = np.array([x_idx, shape[1] / 2, shape[2] / 2, 1])
            x_world = np.dot(affine, x_vox)[:3]
            slice_positions["sagittal"] = {
                "position": float(x_world[0]),
                "axis": "x",
                "bounds": {
                    "minY": float(world_min[1]),
                    "maxY": float(world_max[1]),
                    "minZ": float(world_min[2]),
                    "maxZ": float(world_max[2]),
                },
            }

            # Add volume center for proper positioning
            slice_positions["volume_center"] = {
                "x": float((world_min[0] + world_max[0]) / 2),
                "y": float((world_min[1] + world_max[1]) / 2),
                "z": float((world_min[2] + world_max[2]) / 2),
            }

            # Add full volume bounds for consistent slice sizing
            slice_positions["fullBounds"] = {
                "minX": float(world_min[0]),
                "maxX": float(world_max[0]),
                "minY": float(world_min[1]),
                "maxY": float(world_max[1]),
                "minZ": float(world_min[2]),
                "maxZ": float(world_max[2]),
            }

        # Capture each 2D view
        views = [
            ("axial", vtk_panel.axial_scene),
            ("coronal", vtk_panel.coronal_scene),
            ("sagittal", vtk_panel.sagittal_scene),
        ]

        for name, scene in views:
            if scene is None:
                continue

            try:
                # Render to array
                render_window = scene.GetRenderWindow()
                if render_window is None:
                    continue

                render_window.Render()

                # Get image from render window
                import vtk
                from vtk.util import numpy_support

                w2i = vtk.vtkWindowToImageFilter()
                w2i.SetInput(render_window)
                w2i.ReadFrontBufferOff()
                w2i.Update()

                vtk_image = w2i.GetOutput()
                dims = vtk_image.GetDimensions()

                if dims[0] == 0 or dims[1] == 0:
                    continue

                # Convert to numpy
                scalars = vtk_image.GetPointData().GetScalars()
                if scalars is None:
                    continue

                arr = numpy_support.vtk_to_numpy(scalars)
                n_components = vtk_image.GetNumberOfScalarComponents()
                arr = arr.reshape(dims[1], dims[0], n_components)
                arr = np.flipud(arr)

                # Convert to PIL Image
                if n_components == 4:
                    img = Image.fromarray(arr, mode="RGBA")
                elif n_components == 3:
                    img = Image.fromarray(arr, mode="RGB")
                else:
                    continue

                # Resize if too large
                max_size = 512
                if img.width > max_size or img.height > max_size:
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

                # Convert to JPEG base64
                buffer = io.BytesIO()
                img.convert("RGB").save(buffer, format="JPEG", quality=quality)
                b64_data = base64.b64encode(buffer.getvalue()).decode("ascii")
                slices[name] = f"data:image/jpeg;base64,{b64_data}"

            except Exception as e:
                logger.warning(f"Failed to capture {name} slice: {e}")

    except ImportError:
        logger.warning("PIL not available, skipping slice capture.")

    return slices, slice_positions


def _collect_roi_data(
    main_window: "MainWindow", options: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Collects ROI visualization data.

    Returns:
        List of ROI data dictionaries with simplified geometry.
    """
    rois = []

    for roi_name, roi_layer in main_window.roi_layers.items():
        try:
            roi_data = roi_layer["data"]
            roi_affine = roi_layer["affine"]

            # Get ROI color
            color = main_window.roi_colors.get(roi_name, (1.0, 0.0, 0.0))
            color_rgb = [int(c * 255) for c in color[:3]]

            # Check if it's a sphere ROI
            if (
                main_window.vtk_panel
                and hasattr(main_window.vtk_panel, "sphere_params_per_roi")
                and roi_name in main_window.vtk_panel.sphere_params_per_roi
            ):
                params = main_window.vtk_panel.sphere_params_per_roi[roi_name]
                rois.append(
                    {
                        "name": roi_name,
                        "type": "sphere",
                        "center": params["center"].tolist(),
                        "radius": params["radius"],
                        "color": color_rgb,
                    }
                )

            # Check if it's a rectangle ROI
            elif (
                main_window.vtk_panel
                and hasattr(main_window.vtk_panel, "rectangle_params_per_roi")
                and roi_name in main_window.vtk_panel.rectangle_params_per_roi
            ):
                params = main_window.vtk_panel.rectangle_params_per_roi[roi_name]
                rois.append(
                    {
                        "name": roi_name,
                        "type": "box",
                        "start": np.array(params["start"]).tolist(),
                        "end": np.array(params["end"]).tolist(),
                        "color": color_rgb,
                    }
                )

            else:
                # Generic ROI - compute center of mass
                if np.any(roi_data > 0):
                    coords = np.argwhere(roi_data > 0)
                    center_vox = np.mean(coords, axis=0)
                    # Transform to world coordinates
                    center_world = np.dot(roi_affine, np.append(center_vox, 1.0))[:3]
                    min_coords = np.min(coords, axis=0)
                    max_coords = np.max(coords, axis=0)
                    size = max_coords - min_coords
                    avg_radius = np.mean(size) / 2

                    rois.append(
                        {
                            "name": roi_name,
                            "type": "sphere",
                            "center": center_world.tolist(),
                            "radius": float(avg_radius),
                            "color": color_rgb,
                        }
                    )

        except Exception as e:
            logger.warning(f"Failed to export ROI {roi_name}: {e}")

    return rois


def _generate_html(data: Dict[str, Any], options: Dict[str, Any]) -> str:
    """
    Generates the complete HTML file with embedded three.js visualization.

    Returns:
        Complete HTML content as a string.
    """
    # Compress streamline data
    streamlines_json = json.dumps(data["streamlines"])
    colors_json = json.dumps(data["streamline_colors"])
    rois_json = json.dumps(data["rois"])
    slices_json = json.dumps(data["slices"])
    slice_positions_json = json.dumps(data.get("slice_positions", {}))

    # Optionally compress large data
    if len(streamlines_json) > 100000:
        # Use gzip compression and base64 encoding
        compressed = gzip.compress(streamlines_json.encode("utf-8"))
        streamlines_data = f'"{base64.b64encode(compressed).decode("ascii")}"'
        use_compression = True
    else:
        streamlines_data = streamlines_json
        use_compression = False

    ## TODO - this will be refactored a bit ##
    # Generate HTML content
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TractEdit</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            overflow: hidden;
        }}
        #container {{
            width: 100vw;
            height: 100vh;
            display: flex;
        }}
        #viewer {{
            flex: 1;
            position: relative;
        }}
        #sidebar {{
            width: 280px;
            background: #16213e;
            padding: 20px;
            overflow-y: auto;
            border-left: 1px solid #0f3460;
        }}
        h1 {{
            font-size: 1.2em;
            margin-bottom: 20px;
            color: #e94560;
        }}
        .section {{
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid #0f3460;
        }}
        .section h2 {{
            font-size: 0.9em;
            color: #888;
            margin-bottom: 10px;
            text-transform: uppercase;
        }}
        label {{
            display: flex;
            align-items: center;
            margin-bottom: 8px;
            cursor: pointer;
        }}
        input[type="checkbox"] {{
            margin-right: 10px;
        }}
        input[type="range"] {{
            width: 100%;
            margin: 5px 0;
        }}
        .info {{
            font-size: 0.8em;
            color: #666;
            margin-top: 10px;
        }}
        #slices {{
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}
        #slices img {{
            width: 100%;
            border-radius: 5px;
            border: 1px solid #0f3460;
        }}
        .slice-label {{
            font-size: 0.75em;
            color: #888;
            text-transform: uppercase;
        }}
        #loading {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            text-align: center;
        }}
        .spinner {{
            width: 40px;
            height: 40px;
            border: 3px solid #0f3460;
            border-top-color: #e94560;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }}
        @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
    </style>
</head>
<body>
    <div id="container">
        <div id="viewer">
            <div id="loading">
                <div class="spinner"></div>
                <div>Loading visualization...</div>
            </div>
        </div>
        <div id="sidebar">
            <h1>🧠 TractEdit </h1>
            
            <div class="section">
                <h2>Display</h2>
                <label>
                    <input type="checkbox" id="showStreamlines" checked>
                    Show Streamlines
                </label>
                <label>
                    <input type="checkbox" id="showROIs" checked>
                    Show ROIs
                </label>

            </div>

            
            <div class="section">
                <h2>Streamline Opacity</h2>
                <input type="range" id="opacitySlider" min="0" max="100" value="80">
            </div>
            
            <div class="section" id="slicesSection">
                <h2>Orthogonal Slices</h2>
                <div id="slices"></div>
            </div>
            
            <div class="info">
                <strong>Controls:</strong><br>
                Left-drag: Rotate<br>
                Right-drag: Pan<br>
                Scroll: Zoom
            </div>
            
            <div class="info" style="margin-top: 20px">
                Exported from TractEdit
            </div>
        </div>
    </div>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script>
        // Data
        const COMPRESSED = {'true' if use_compression else 'false'};
        const streamlinesData = {streamlines_data};
        const colorsData = {colors_json};
        const roisData = {rois_json};
        const slicesData = {slices_json};
        const slicePositionsData = {slice_positions_json};
        
        // Decompress if needed
        async function decompressData(b64data) {{
            const compressed = Uint8Array.from(atob(b64data), c => c.charCodeAt(0));
            const ds = new DecompressionStream('gzip');
            const stream = new Blob([compressed]).stream().pipeThrough(ds);
            const text = await new Response(stream).text();
            return JSON.parse(text);
        }}
        
        // Three.js setup
        let scene, camera, renderer, controls;
        let streamlineGroup, roiGroup;
        
        async function init() {{
            // Get streamlines data
            let streamlines;
            if (COMPRESSED) {{

                streamlines = await decompressData(streamlinesData);
            }} else {{
                streamlines = streamlinesData;
            }}
            
            // Scene
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x1a1a2e);
            
            // Camera
            const viewer = document.getElementById('viewer');
            camera = new THREE.PerspectiveCamera(
                60, viewer.clientWidth / viewer.clientHeight, 0.1, 10000
            );
            
            // Compute center and size from streamlines for rotation
            let centerX, centerY, centerZ, size;
            
            // Always use streamline bounds for rotation center
            let minX = Infinity, minY = Infinity, minZ = Infinity;
            let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
            
            for (const sl of streamlines) {{
                for (const pt of sl) {{
                    minX = Math.min(minX, pt[0]);
                    minY = Math.min(minY, pt[1]);
                    minZ = Math.min(minZ, pt[2]);
                    maxX = Math.max(maxX, pt[0]);
                    maxY = Math.max(maxY, pt[1]);
                    maxZ = Math.max(maxZ, pt[2]);
                }}
            }}
            
            centerX = (minX + maxX) / 2;
            centerY = (minY + maxY) / 2;
            centerZ = (minZ + maxZ) / 2;
            size = Math.max(maxX - minX, maxY - minY, maxZ - minZ);
            
            // Use streamline center for rotation target
            let targetX = centerX, targetY = centerY, targetZ = centerZ;
            
            // Set camera up vector to Z (medical imaging RAS convention)
            camera.up.set(0, 0, 1);
            camera.position.set(targetX, targetY - size * 1.5, targetZ + size * 0.5);
            camera.lookAt(targetX, targetY, targetZ);
            
            // Renderer
            renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(viewer.clientWidth, viewer.clientHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            viewer.appendChild(renderer.domElement);
            
            // Controls
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.target.set(targetX, targetY, targetZ);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.update();
            
            // Lighting
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
            scene.add(ambientLight);
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.4);
            directionalLight.position.set(1, 1, 1);
            scene.add(directionalLight);
            
            // Create streamlines
            streamlineGroup = new THREE.Group();
            
            for (let i = 0; i < streamlines.length; i++) {{
                const sl = streamlines[i];
                const color = colorsData[i] || [200, 200, 200];
                
                const geometry = new THREE.BufferGeometry();
                const positions = new Float32Array(sl.length * 3);
                
                for (let j = 0; j < sl.length; j++) {{
                    positions[j * 3] = sl[j][0];
                    positions[j * 3 + 1] = sl[j][1];
                    positions[j * 3 + 2] = sl[j][2];
                }}
                
                geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                
                const material = new THREE.LineBasicMaterial({{
                    color: new THREE.Color(color[0]/255, color[1]/255, color[2]/255),
                    transparent: true,
                    opacity: 0.8
                }});
                
                const line = new THREE.Line(geometry, material);
                streamlineGroup.add(line);
            }}
            
            scene.add(streamlineGroup);
            
            // Create ROIs
            roiGroup = new THREE.Group();
            
            for (const roi of roisData) {{
                const color = new THREE.Color(roi.color[0]/255, roi.color[1]/255, roi.color[2]/255);
                
                if (roi.type === 'sphere') {{
                    const geometry = new THREE.SphereGeometry(roi.radius, 16, 16);
                    const material = new THREE.MeshPhongMaterial({{
                        color: color,
                        transparent: true,
                        opacity: 0.5,
                        side: THREE.DoubleSide
                    }});
                    const mesh = new THREE.Mesh(geometry, material);
                    mesh.position.set(roi.center[0], roi.center[1], roi.center[2]);
                    roiGroup.add(mesh);
                }} else if (roi.type === 'box') {{
                    const size = [
                        Math.abs(roi.end[0] - roi.start[0]),
                        Math.abs(roi.end[1] - roi.start[1]),
                        Math.abs(roi.end[2] - roi.start[2])
                    ];
                    const center = [
                        (roi.start[0] + roi.end[0]) / 2,
                        (roi.start[1] + roi.end[1]) / 2,
                        (roi.start[2] + roi.end[2]) / 2
                    ];
                    const geometry = new THREE.BoxGeometry(size[0], size[1], size[2]);
                    const material = new THREE.MeshPhongMaterial({{
                        color: color,
                        transparent: true,
                        opacity: 0.5,
                        side: THREE.DoubleSide
                    }});
                    const mesh = new THREE.Mesh(geometry, material);
                    mesh.position.set(center[0], center[1], center[2]);
                    roiGroup.add(mesh);
                }}
            }}
            
            scene.add(roiGroup);
            
            
            // Add slices to sidebar
            const slicesContainer = document.getElementById('slices');
            for (const [name, src] of Object.entries(slicesData)) {{
                const label = document.createElement('div');
                label.className = 'slice-label';
                label.textContent = name;
                slicesContainer.appendChild(label);
                
                const img = document.createElement('img');
                img.src = src;
                img.alt = name + ' slice';
                slicesContainer.appendChild(img);
            }}
            
            // Hide loading
            document.getElementById('loading').style.display = 'none';
            
            // Controls
            document.getElementById('showStreamlines').addEventListener('change', (e) => {{
                streamlineGroup.visible = e.target.checked;
            }});
            
            document.getElementById('showROIs').addEventListener('change', (e) => {{
                roiGroup.visible = e.target.checked;
            }});
            

            
            document.getElementById('opacitySlider').addEventListener('input', (e) => {{
                const opacity = e.target.value / 100;
                streamlineGroup.children.forEach(line => {{
                    line.material.opacity = opacity;
                }});
            }});

            
            // Handle resize
            window.addEventListener('resize', () => {{
                camera.aspect = viewer.clientWidth / viewer.clientHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(viewer.clientWidth, viewer.clientHeight);
            }});
            
            // Animation loop
            function animate() {{
                requestAnimationFrame(animate);
                controls.update();
                renderer.render(scene, camera);
            }}
            animate();
        }}
        
        init();
    </script>
</body>
</html>"""

    return html
