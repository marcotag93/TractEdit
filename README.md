<p align="center">
  <img src="tractedit_pkg/assets/logo.png" alt="TractEdit Logo" width="150"/>
</p>

<h1 align="center">🧠 TractEdit</h1>

<p align="center">
  <b>A Python-based Open-Source Interactive Tool for Virtual Dissection and Manual Refinement of Diffusion MRI Tractography</b>
</p>

<p align="center">
  <code>.trk</code> • <code>.tck</code> • <code>.trx</code> • <code>.vtk</code> • <code>.vtp</code>
</p>

<p align="center">
  <a href="https://doi.org/10.1088/1741-2552/ae9346"><img src="https://img.shields.io/badge/Paper_DOI-10.1088%2F1741--2552%2Fae9346-B31B1B" alt="Journal of Neural Engineering paper DOI"></a>
  <a href="https://doi.org/10.5281/zenodo.18118153"><img src="https://img.shields.io/badge/Zenodo_DOI-10.5281%2Fzenodo.18118153-1682D4" alt="Zenodo software archive DOI"></a>
  <a href="https://github.com/marcotag93/TractEdit/releases/latest"><img src="https://img.shields.io/github/v/release/marcotag93/TractEdit" alt="Latest release"></a>
  <a href="https://github.com/marcotag93/TractEdit/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"/></a>
  <img src="https://img.shields.io/badge/python-3.11-green.svg" alt="Python"/>
  <a href="https://github.com/marcotag93/TractEdit/releases/latest"><img src="https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey" alt="Platform — download the latest release"/></a>
</p>

> [!IMPORTANT]
> **Research Use Only.** TractEdit is a research tool for the visualization, virtual dissection, and quality control of diffusion MRI tractography. It is **not a medical device**, has not been clinically validated, and holds no regulatory clearance (FDA / CE). It must not be used for diagnosis, treatment planning, or neurosurgical guidance.


https://github.com/user-attachments/assets/8b95cfea-a0b9-4537-8e0f-0ac645f44294


---

## 👤 Author

**Marco Tagliaferri** — *PhD Candidate in Neuroscience*
🏛️ [Center for Mind/Brain Sciences (CIMeC)](https://www.cimec.unitn.it/), University of Trento, Italy

[![Email](https://img.shields.io/badge/Email-marco.tagliaferri%40unitn.it-D14836?style=flat&logo=gmail&logoColor=white)](mailto:marco.tagliaferri@unitn.it)
[![Email](https://img.shields.io/badge/Email-marco.tagliaferri93%40gmail.com-D14836?style=flat&logo=gmail&logoColor=white)](mailto:marco.tagliaferri93@gmail.com)
[![ORCID](https://img.shields.io/badge/ORCID-0000--0002--1800--3977-A6CE39?style=flat&logo=orcid&logoColor=white)](https://orcid.org/0000-0002-1800-3977)
[![GitHub](https://img.shields.io/badge/GitHub-marcotag93-181717?style=flat&logo=github)](https://github.com/marcotag93)

If you use TractEdit in your research, please cite:

> Tagliaferri, M., & Cattaneo, L. (2026). TractEdit: An Open-Source Interactive Tool for Virtual Dissection and Manual Refinement of Diffusion MRI Tractography. *Journal of Neural Engineering*, 23(4), 046046. https://doi.org/10.1088/1741-2552/ae9346

**BibTeX:**

```bibtex
@article{Tagliaferri_2026,
  author = {Tagliaferri, Marco and Cattaneo, Luigi},
  title = {{TractEdit}: an open-source interactive tool for virtual dissection and manual refinement of diffusion {MRI} tractography},
  journal = {Journal of Neural Engineering},
  publisher = {IOP Publishing},
  year = {2026},
  month = {aug},
  volume = {23},
  number = {4},
  pages = {046046},
  issn = {1741-2552},
  doi = {10.1088/1741-2552/ae9346},
  url = {https://doi.org/10.1088/1741-2552/ae9346}
}
```

---

## 📋 Table of Contents

- [Key Features](#key-features)
- [Getting Started](#getting-started-manual-install)
- [Keyboard Shortcuts](#keyboard-shortcuts)
- [Sample Workflow](#sample-workflow)
- [Author](#-author)
- [License](#license)

## ✨ Key Features

### 📂 File I/O

Load & save streamlines in `.trk`, `.tck`, `.trx`, `.vtk`, `.vtp` formats with whole-brain support (>5M streamlines)

### 🖥️ Visualization

- Multi-view: 3D + Axial, Coronal, Sagittal slices
- Anatomical overlay with NIfTI support and native-resolution oblique-image display
- RGB, scalar, or greyscale coloring
- Line or tube rendering

### ✏️ ROI & Editing Tools

- **Drawing:** Pencil, Eraser, Sphere, Rectangle modes
- **Selection:** Sphere-based streamlines selection and deletion with adjustable radius
- **Filtering:** Include/Exclude logic for ROIs
- Undo/Redo support for streamline deletions and ROI operations

### 🧠 Advanced Analysis

- FreeSurfer parcellation support with connectivity matrices
- ODF 3D visualization (spherical harmonics)
- Track Density Imaging (TDI)
- Centroid & Medoid calculation

<details>
<summary><b>📋 Full Feature List</b></summary>

#### File Support

- **Load & Save** streamline bundles (`.trk`, `.tck`, `.trx`, `.vtk`, `.vtp`)
- **Whole-Brain Tractogram Support:** Optimized rendering for large datasets (tested with >5 million streamlines) using stride-based visualization and toggleable "skip"

#### Visualization

- **Multi-View Orthogonal Visualization:** Integrated 3D viewer and three linked 2D orthogonal slice views (Axial, Coronal, Sagittal)
- **Anatomical Image:** Load NIfTI images (`.nii`, `.nii.gz`) for anatomical context and interactive slice navigation
  - Oblique acquisitions are displayed on their canonical native voxel grid without interpolation, preserving source resolution. For scanner-RAS co-registration, use already-conformed inputs.
- **3D Visualization** with [VTK](https://vtk.org/) and [FURY](https://fury.gl/)
  - Default orientation (RGB), or scalar-based coloring with dynamic colormap range adjustment, or greyscale
  - **Render as Lines or Tubes:** Toggle between fast line rendering and high-quality 3D tube rendering via **View → Streamline Geometry**
  - Interactive RAS coordinate bar for precise navigation

#### ROI Support

- **Multi-Layer Anatomical ROI Support:** Load multiple NIfTI images (`.nii`, `.nii.gz`) as Region of Interest (ROI) layers
  - Independent visibility toggles and matching 2D/3D color settings
  - **Logical Filtering:** Right-click ROIs to set them as exclusion or inclusion ROIs for streamlines

#### Interactive ROI Drawing Tools

- **Pencil Mode (1):** Freehand drawing directly on 2D slice views to create custom ROIs
- **Eraser Mode (2):** Erase portions of ROIs with freehand strokes
- **Sphere Mode (3):** Draw spherical ROIs on slices
- **Rectangle Mode (4):** Draw rectangular/cuboid ROIs on slices
- **Move ROIs:** Hold Ctrl and drag to reposition sphere or rectangle ROIs with real-time preview. Hold Ctrl + scroll to resize
- **Undo/Redo Support:** Full undo/redo for all ROI drawing operations (Ctrl+Z / Ctrl+Y) if mode enabled, otherwise for streamline deletion

#### Interactive Editing Tools

- Sphere-based streamline selection (with adjustable radius)
- Streamline deletion and undo/redo support
- Screenshot export

#### FreeSurfer Parcellation Support

- Load FreeSurfer parcellation/segmentation files (`aparc+aseg`, `aparc.a2009s+aseg`, etc.)
- **3D Parcellation Overlay:** Visualize connected parcellation regions in 3D with hemisphere-organized tree view
- **Region Logic Filters:** Set parcellation regions as Include/Exclude filters for streamlines
- **Compute Connectivity Matrix:** Generate structural connectivity matrices from streamlines and parcellation

#### ODF Visualization

- Load Spherical Harmonics (SH) coefficient NIfTI files
- **ODF Tunnel View:** Visualize ODFs masked by the current bundle's spatial extent

#### Export Options

- **Track Density Imaging (TDI):** Save density maps of visible streamlines as NIfTI files
- **HTML Export (Experimental):** Export interactive 3D visualization as self-contained HTML file
- Screenshot export in multiple formats

#### Bundle Analytics

- Calculate **Centroid** and **Medoid** (both AOT-compiled) of the edited bundle with cancellable progress and batched distance computation

#### UI & Performance

- **Theme Support:** Light, Dark, and System theme modes with full UI styling
- **Streamline Info Display:** File name, streamline count, voxel size, bounding box, etc. with vertical data panel and hover details
- **Keyboard Shortcuts** for fast interaction (see full list below)
- **Fast Startup:** Splash screen implementation for immediate feedback and optimized library loading
- **Background Loading:** Non-blocking threaded loading for large streamline bundles and anatomical images
- **Memory-Mapped Images:** Efficient on-demand slice extraction for large anatomical images without loading full volume into RAM
- **Modular Architecture:** Refactored codebase with dedicated manager classes (ThemeManager, StateManager, SelectionManager, etc.) for improved maintainability
- **Performance Optimizations:** AOT-compiled numerical kernels with binary search resampling and parallel batch processing for geometric computations, Numpy vectorizations, debounced UI updates, pre-computed bounding boxes for fast selection, TRX bbox cache on reload (`_tractedit_bboxes`), and TRX-native save path (`select(copy_safe=True)` + `tbx.save()`) to avoid nibabel round-trips
- **Reliability:** Comprehensive automated test suite (152 tests) ensuring stability of core features

#### 💡 Tips for Large Datasets

*📌 While TractEdit supports **whole-brain tractograms**, rendering density may be automatically reduced for extremely large files to maintain interactivity. You can adjust this manually in the toolbar.*

**Selection Strategy:**

| Method                           | Best For                                            |
| -------------------------------- | --------------------------------------------------- |
| **Sphere Selection**       | Refining specific bundles or small bundle complexes |
| **ROI Filtering**          | Whole-brain tractograms (works on entire dataset)   |
| **ROI Drawing**            | Custom regions on anatomical slices                 |
| **Parcellation Filtering** | Anatomically-guided filtering with FreeSurfer       |

</details>

---

## Getting Started (Manual Install)

### 1. Clone the Repository

```bash
git clone https://github.com/marcotag93/TractEdit.git
cd TractEdit
```

### 2. Install Dependencies

The project dependencies (including PyQt6, VTK, and Nibabel) are defined in pyproject.toml

- **Python 3.11** (required)
- [PyQt6](https://pypi.org/project/PyQt6/)
- [VTK](https://vtk.org/)
- [FURY](https://fury.gl/)
- [Nibabel](https://nipy.org/nibabel/)
- NumPy
- [trx-python](https://pypi.org/project/trx-python/)
- Scipy

> **Note:** Python 3.11 is the only officially supported version. Other Python versions may experience compatibility issues with dependency combinations.

Recommend a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install the app and its dependencies:
pip install .
```

> **Note:** The install process automatically compiles the AOT numerical extension (optimized computational kernels) for your operating system. [Numba](https://numba.pydata.org/) is temporarily downloaded as a build-time dependency to compile the extension, but is **not** installed into your environment — only the compiled binary is kept.

### 3. Launch the App

The application can now be launched using the tractedit command installed via pip.

```bash
tractedit
```


#### Command Line Options

```bash
# Load files directly
tractedit bundle.trk --anat T1w.nii.gz

# Load multiple ROI files at startup
tractedit bundle.trk --anat T1w.nii.gz --load-roi roi1.nii.gz --load-roi roi2.nii.gz

# Load file creating a spherical ROI at RAS coordinates
tractedit bundle.trk --anat T1w.nii.gz --roi 10,20,30 --radius 5

# Headless format conversion (no GUI)
tractedit input.trk --convert-to output.trx

# Headless density map export (TDI)
tractedit input.trk --density-map output.nii.gz --anat T1w.nii.gz

# Display version and help
tractedit --version
tractedit --help
```

> On certain Linux systems (e.g., Ubuntu on Wayland), you may encounter Qt platform errors. If so, try the xcb workaround below.

```bash
QT_QPA_PLATFORM=xcb tractedit
```

### 4. Load Sample Data (Optional)

Explore `sample_data/` to test TractEdit with example streamline files, anatomical file, ROI files, parcellation file and ODF files.

---

### For Developers / Contributors

If you want to contribute or build from source, install the full development stack:

```bash
# Install all dependency groups
poetry install --with build,dev,test

# Build the AOT-compiled numerical extension for your OS
# (this step is automatic when using `pip install .`, but
# must be run manually after `poetry install`)
python tractedit_pkg/_numba_aot/build_aot.py

# Run the app
poetry run tractedit

# Run tests
poetry run pytest tests/ -v
```

> **Note:** The AOT extension produces a platform-specific binary (`.pyd` on Windows, `.so` on Linux/macOS) and must be rebuilt after modifying any functions in `tractedit_pkg/_numba_aot/build_aot.py`. These compiled files are not tracked in git — each developer builds them locally.

### Pre-built Executables

No Python setup is required for these versions. Download the latest [release](https://github.com/marcotag93/TractEdit/releases) for your operating system:

* **Windows:** Use the `.exe` file.
* **macOS (Apple Silicon):** Use the `.dmg` file.
* **Linux (AppImage):** Use `.AppImage` — portable, runs on most Linux distributions without installation. Simply make it executable (`chmod +x`) and run.
* **Linux (Debian/Ubuntu):** Use `.deb` — native package for Debian-based distributions. Install with `sudo dpkg -i TractEdit_3.4.7_amd64.deb`.


---

## Keyboard Shortcuts

| Key / Combo                 | Action                                                        |
| --------------------------- | ------------------------------------------------------------- |
| **s**                 | Add streamlines at cursor to selection (selection grows only)   |
| **Shift+s**           | Remove streamlines at cursor from selection (selection shrinks only) |
| **i**                 | Invert selection                                              |
| **d**                 | Delete selected streamlines                                   |
| **c**                 | Clear current selection                                       |
| **+ / =**             | Increase selection sphere radius                              |
| **-**                 | Decrease selection sphere radius                              |
| **↑ / ↓**           | Axial Slice navigation (Z-axis)                               |
| **← / →**           | Sagittal Slice navigation (X-axis)                            |
| **Shift+Scroll**      | Slice navigation on the 2D panel under cursor                 |
| **1**                 | Toggle Pencil drawing mode                                    |
| **2**                 | Toggle Eraser drawing mode                                    |
| **3**                 | Toggle Sphere ROI drawing mode                                |
| **4**                 | Toggle Rectangle ROI drawing mode                             |
| **Ctrl+↑ / Ctrl+↓** | Coronal Slice navigation (Y-axis)                             |
| **Ctrl+Click**        | Replace sphere/rectangle ROI (when in mode)                   |
| **Ctrl+Drag**         | Move sphere/rectangle ROI (when in mode)                      |
| **Ctrl+Scroll**       | Resize sphere/rectangle ROI (when in mode)                    |
| **Ctrl+s**            | Save As                                                       |
| **Ctrl+z**            | Undo last deletion / ROI operation                           |
| **Ctrl+y / Shift+z**  | Redo last undone deletion / ROI operation                    |
| **Ctrl+p**            | Save a screenshot                                             |
| **Esc**               | Hide selection sphere                                         |
| **Ctrl+q**            | Quit application                                              |

---

## Sample Workflow

### Step 1: Load Your Data

```
File → Load          → Load .trk, .tck, .trx, .vtk, or .vtp
File → Load Image    → Add anatomical image (NIfTI)
```

### Step 2: Navigate

- **2D Slices:** Click-drag, arrow keys, or Shift+scroll on any 2D panel
- **3D View:** Rotate, zoom, pan with mouse

### Step 3: Edit (Choose Your Approach)

<table>
<tr>
<td width="25%" align="center">

**🎯 Manual Selection**

`+`/`-` adjust radius <br>
`S` to add streamlines <br>
`Shift+S` to remove streamlines <br>
`I` to invert selection <br>
`D` to delete

</td>
<td width="25%" align="center">

**📂 Load ROIs**

File → Load ROI  
Right-click for  
Include/Exclude  

</td>
<td width="25%" align="center">

**✏️ Draw ROIs**

`1` Pencil · `2` Eraser  
`3` Sphere · `4` Rectangle  
Ctrl+drag to move 
Include/Exclude 

</td>
<td width="25%" align="center">

**🧠 Parcellation**

File → Load Parcellation  
Right-click regions  
Include/Exclude  

</td>
</tr>
</table>


### Step 4: Finalize & Export

| Action                    | Menu                             |
| ------------------------- | -------------------------------- |
| Change colors             | View → Streamline Color          |
| Calculate centroid/medoid | File → Calculate Centroid/Medoid |
| Save density map          | File → Save Density Map          |
| Export HTML               | File → Export to HTML            |
| Save bundle               | File → Save As                   |

*💡 Tip: Use `Ctrl+Z` / `Ctrl+Y` for undo/redo at any time!*

---

## License

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
