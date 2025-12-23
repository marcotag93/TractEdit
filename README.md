# TractEdit

A Python-based graphical interface for interactively **viewing**, **selecting**, and **editing** tractography bundles in `.trk`, `.tck`, `.trx`, `.vtk` and `.vtp` formats.








https://github.com/user-attachments/assets/2101fb2e-1e3b-487b-a77b-fd7ba45b5d65






---

## Author

**Marco Tagliaferri**, PhD Candidate in Neuroscience  
Center for Mind/Brain Sciences (CIMeC), University of Trento Italy 

📧 marco.tagliaferri@unitn.it / marco.tagliaferri93@gmail.com

---

## Key Features

- **Load & Save** streamline bundles (`.trk`, `.tck`, `.trx`, `.vtk`, `.vtp`)
- **Whole-Brain Tractogram Support:** Optimized rendering for large datasets (tested with >2 million streamlines) using stride-based visualization and toggleable "skip".
- **Multi-View Orthogonal Visualization:** Integrated 3D viewer and three linked 2D orthogonal slice views (Axial, Coronal, Sagittal).
- **Anatomical Image:** Load NIfTI images (`.nii`, `.nii.gz`) as backgrounds for anatomical context and **interactive slice navigation**.
- **Multi-Layer Anatomical ROI Support:** - Load multiple NIfTI images (`.nii`, `.nii.gz`) as background Region of Interest (ROI) layers.
    - Independent visibility toggles and color settings.
    - **Logical Filtering:** Right-click ROIs to set them as exclusion or inclusion ROIs for streamlines.
- **Interactive ROI Drawing Tools:**
    - **Pencil Mode (1):** Freehand drawing directly on 2D slice views to create custom ROIs.
    - **Eraser Mode (2):** Erase portions of ROIs with freehand strokes.
    - **Sphere Mode (3):** Draw spherical ROIs on slices.
    - **Rectangle Mode (4):** Draw rectangular/cuboid ROIs on slices.
    - **Move ROIs:** Hold Ctrl and drag to reposition sphere or rectangle ROIs with real-time preview. Hold Ctrl + scroll to resize.
    - **Undo/Redo Support:** Full undo/redo for all ROI drawing operations (Ctrl+Z / Ctrl+Y) if mode enabled, otherwise for streamline deletion.
- **3D Visualization** with [VTK](https://vtk.org/) and [FURY](https://fury.gl/)
    - Default orientation (RGB), or scalar-based coloring with dynamic colormap range adjustment, or greyscale.
    - **Render as Lines or Tubes:** Toggle between fast line rendering and high-quality 3D tube rendering via **View → Streamline Geometry**.
    - Interactive RAS coordinate bar for precise navigation.
- **Interactive Editing Tools:**
    - Sphere-based streamline selection (with adjustable radius)
    - Streamline deletion and undo/redo support.
    - Screenshot export
- **FreeSurfer Parcellation Support:**
    - Load FreeSurfer parcellation/segmentation files (`aparc+aseg`, `aparc.a2009s+aseg`, etc.)
    - **3D Parcellation Overlay:** Visualize connected parcellation regions in 3D with hemisphere-organized tree view
    - **Region Logic Filters:** Set parcellation regions as Include/Exclude filters for streamlines
    - **Compute Connectivity Matrix:** Generate structural connectivity matrices from streamlines and parcellation
- **ODF Visualization:**
    - Load Spherical Harmonics (SH) coefficient NIfTI files
    - **ODF Tunnel View:** Visualize ODFs masked by the current bundle's spatial extent
- **Track Density Imaging (TDI):** Save density maps of visible streamlines as NIfTI files.
- **Export Options:**
    - **HTML Export (Experimental):** Export interactive 3D visualization as self-contained HTML file
    - Screenshot export in multiple formats
- **Bundle Analytics:**
    - Calculate **Centroid** and **Medoid** (both Numba optimized) of the edited bundle.
- **Streamline Info Display:**
    - File name, streamline count, voxel size, bounding box, etc.
    - Vertical data panel with hover details 
- **Keyboard Shortcuts** for fast interaction (see full list below)
- **Fast Startup:** Splash screen implementation for immediate feedback and optimized library loading.
- **Modular Architecture:** Refactored codebase with dedicated manager classes for improved maintainability.
- **Performance Optimizations:** Numba JIT compilation for geometric computations, Numpy vectorizations, debounced UI updates, pre-computed bounding boxes for fast selection.

> ⚠️ **Note:** While TractEdit supports **whole-brain tractograms**, rendering density may be automatically reduced (skipped) for extremely large files to maintain interactivity. You can adjust this manually in the toolbar.
>
> **Selection Strategy:**
> * **Sphere Selection:** Recommended for refining specific **bundles** or small bundle complexes. Note that this method is **ineffective** on whole-brain datasets if streamlines are being "skipped" (hidden) for performance, as the sphere can only interact with visible fibers.
> * **ROI Filtering:** Recommended for **whole-brain tractograms**. Loading NIfTI images as logic filters (Include/Exclude) works on the entire dataset regardless of visual density.
> * **ROI Drawing:** Use the built-in drawing tools (Pencil, Sphere, Rectangle) to create custom ROIs directly on anatomical slices for flexible, targeted filtering.
> * **Parcellation Filtering:** Load a FreeSurfer parcellation and set regions as Include/Exclude for anatomically-guided filtering.
---

## Getting Started (Manual Install)

### 1. Clone the Repository
```bash
git clone https://github.com/marcotag93/TractEdit.git
cd TractEdit
```

### 2. Install Dependencies
The project dependencies (including PyQt6, VTK, and Nibabel) are defined in pyproject.toml

- Python 3.8–3.11 (tested)
- [PyQt6](https://pypi.org/project/PyQt6/)
- [VTK](https://vtk.org/)
- [FURY](https://fury.gl/)
- [Nibabel](https://nipy.org/nibabel/)
- NumPy
- [Numba](https://numba.pydata.org/)
- pytz
- [trx-python](https://pypi.org/project/trx-python/)
  
Recommend a virtual environment:
```bash
python -m venv venv
source venv/bin/activate 
# Install the app and its dependencies:
pip install .
```

### 3. Launch the App
The application can now be launched using the tractedit command installed via pip.
```bash
tractedit
```

#### Command Line Options
```bash
# Load files directly
tractedit bundle.trk --anat T1w.nii.gz

# Load file creating a spherical ROI at RAS coordinates
tractedit bundle.trk --anat T1w.nii.gz --roi 10,20,30 --radius 5

# Headless format conversion (no GUI)
tractedit input.trk --convert-to output.trx

# Display version and help
tractedit --version
tractedit --help
```

> On certain Linux systems (e.g., Ubuntu on Wayland), you may encounter Qt platform errors. If so, try the xcb workaround below. Please note that this mode is known to cause rendering artifacts in the 2D panels.
```bash
QT_QPA_PLATFORM=xcb tractedit
```

### 4. Load Sample Data (Optional)
Explore `sample_data/` to test TractEdit with example streamline files, anatomical file, ROI files, parcellation file and ODF files.

---

### For Windows
Use **pre-built executable** Tractedit.exe (no Python setup required).

---

## Keyboard Shortcuts

| Key / Combo          | Action                                      |
|----------------------|---------------------------------------------|
| **s**                | Select/Deselect streamlines at cursor       |
| **c**                | Clear current selection                     |
| **d**                | Delete selected streamlines                 |
| **+ / =**            | Increase selection sphere radius            |
| **-**                | Decrease selection sphere radius            |
| **↑ / ↓**            | Axial Slice navigation (Z-axis)             |
| **← / →**            | Sagittal Slice navigation (X-axis)          |
| **1**                | Toggle Pencil drawing mode                  |
| **2**                | Toggle Eraser drawing mode                  |
| **3**                | Toggle Sphere ROI drawing mode              |
| **4**                | Toggle Rectangle ROI drawing mode           |
| **Ctrl+↑ / Ctrl+↓**  | Coronal Slice navigation (Y-axis)           |
| **Ctrl+Click**       | Replace sphere/rectangle ROI (when in mode) |
| **Ctrl+Drag**        | Move sphere/rectangle ROI (when in mode)    |
| **Ctrl+Scroll**      | Resize sphere/rectangle ROI (when in mode)  |
| **Ctrl+s**           | Save As                                     |
| **Ctrl+z**           | Undo last deletion / ROI operation          |
| **Ctrl+y / Shift+z** | Redo last undone deletion / ROI operation   |
| **Ctrl+p**           | Save a screenshot                           |
| **Esc**              | Hide selection sphere                       |
| **Ctrl+q**           | Quit application                            |

---

## Sample Workflow

1. Open a `.trk`, `.tck`, `.trx`, `.vtk` or `.vtp` file via **File → Open**
2. Load an anatomical image via **File → Load Image** to enable 2D slice views.
3. Use the mouse click-drag in the 2D slice views or the arrow keys (see shortcuts above) to navigate the anatomical slices.

Choose one or more of the following approaches:

**Option A - Manual Selection (Main Feature):**
- Click on the 3D view and press `S` to select streamlines under the cursor
- Use `+` / `-` to adjust the selection sphere radius
- Press `D` to delete selected streamlines, or `C` to clear selection
- Use `Ctrl+Z` to undo deletions, `Ctrl+Y` to redo

**Option B - Load ROIs:**
- Load anatomical ROIs via **File → Load ROI**
- Right-click ROI layers in the data panel to set them as **Include** or **Exclude** regions
- Streamlines are automatically filtered based on ROI logic

**Option C - Draw ROIs:**
- Use the drawing toolbar to create custom ROIs directly on slices:
  - Press **1** for Pencil mode to freehand draw
  - Press **3** for Sphere mode, click to place and drag to resize
  - Press **4** for Rectangle mode for rectangular regions
  - Hold **Ctrl** and drag to move placed sphere/rectangle ROIs
  - Use **2** for Eraser mode to remove parts of ROIs
- Right-click drawn ROIs in the data panel to set them as Include/Exclude filters

**Option D - Parcellation-Based Filtering:**
- Load a FreeSurfer parcellation via **File → Load Parcellation**
- Enable the 3D overlay via **View → Show Parcellation Overlay**
- Expand the parcellation in the data panel to see connected regions by hemisphere
- Right-click regions to set them as **Include** or **Exclude** filters
- Compute a connectivity matrix via **File → Compute Connectivity Matrix**

4. If needed, change streamline color in **View → Streamline Color**. If using Color by Scalar, use the Scalar Range toolbar at the top of the window to adjust the min/max range of the colormap.
5. Save the centroid and/or the medoid of your edited bundle with **File → Calculate Centroid** and **File → Calculate Medoid**
6. Save a Track Density Image with **File → Save Density Map**
7. Export an interactive HTML visualization with **File → Export to HTML** (Experimental)
8. Save your edited bundle with **File → Save As** choosing one of the available formats (`.trk`, `.tck`, `.trx`, `.vtk`, `.vtp`)

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
