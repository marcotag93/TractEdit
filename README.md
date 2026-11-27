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





https://github.com/user-attachments/assets/601a22cb-d96e-4c46-b582-d4cbe1a6d443





<sub>The surface mesh and the TRX file shown with it in this video were obtained from the <a href="https://zenodo.org/records/20681790">Track4Edu repository on Zenodo</a> and used solely for this video demonstration. We thank the Track4Edu authors for making these data available.</sub>

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

- [Key Features](#-key-features)
- [Getting Started](#getting-started)
- [Keyboard Shortcuts](#keyboard-shortcuts)
- [Sample Workflow](#sample-workflow)
- [Surface Meshes](#surface-meshes)
- [Polylines](#polylines)
- [Surface Polygons](#surface-polygons)
- [Save and Restore Sessions](#save-and-restore-sessions)
- [Command Line](#command-line)
- [Contributing and Support](#contributing-and-support)
- [License](#license)

## ✨ Key Features

TractEdit lets you inspect, dissect and refine existing diffusion MRI tractograms in an interactive desktop workspace.

### 📂 File I/O

- **Tractograms:** Load, replace and save bundles in `.trk`, `.tck`, `.trx`, `.vtk` and `.vtp` formats
- **Images & ROIs:** Load and save NIfTI volumes (`.nii`, `.nii.gz`)
- **Sessions:** Save the editing workspace to `.tractedit-session` and reopen it later

### 🖥️ Visualization

- **Multi-view:** 3D + Axial, Coronal and Sagittal slices
- **Display:** Lines or tubes, orientation colours, scalar maps or greyscale
- **Images:** Affine-aware oblique display, 4D volume selection and RGB display

### ✏️ ROI & Editing Tools

- **Drawing:** Pencil, Eraser, Sphere and Rectangle modes
- **Selection:** Sphere-based streamline selection and deletion with adjustable radius
- **Filtering:** Include/Exclude logic for loaded or drawn regions of interest (ROIs)
- **Undo/Redo:** Streamline deletions, ROI edits, polyline and surface polygon edits

### 🧠 Advanced Analysis

- FreeSurfer parcellation filters and connectivity matrices
- Orientation distribution function (ODF) visualization from spherical harmonics
- Track Density Imaging (TDI), centroid and medoid calculation

### 🧩 Surfaces & Annotations

- PLY, VTP and OBJ meshes with independent display controls and OBJ textures
- Draw 3D polylines on surfaces or import ordered traces from CSV or VTP
- Draw, export and import closed surface polygons (`*_polygon.vtp` / `*_patch.vtp`, `*_polygon.obj` / `*_patch.obj`)
- Experimental endpoint proximity and surface-intersection mapping

### 📤 Export & Sharing

- Edited bundles, screenshots and offline HTML views
- Scientific surface-mapping data and figures

*💡 Tip: For large tractograms, use **Skip** to reduce rendering density. ROI filtering still uses the full dataset. Large anatomical images may use a reduced-resolution preview; scientific density outputs retain the reference grid.*

---

## Getting Started

### Pre-built Executables

Get the desktop app for your system. No Python installation is needed. You can also browse all packages and release notes on the [Releases page](https://github.com/marcotag93/TractEdit/releases).

| Download (`v3.6.1`) | Package and launch |
| --- | --- |
| <a href="https://github.com/marcotag93/TractEdit/releases/download/v3.6.1/TractEdit-3.6.1-windows-x64.zip"><img src="https://img.shields.io/badge/Download_for_Windows-ZIP_(x64)-30363d?style=for-the-badge&labelColor=24292f&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0iI2ZmZmZmZiI%2BPHBhdGggZD0iTTAgMGgxMS4zdjExLjNIMHptMTIuNyAwSDI0djExLjNIMTIuN3pNMCAxMi43aDExLjNWMjRIMHptMTIuNyAwSDI0VjI0SDEyLjd6Ii8%2BPC9zdmc%2B" alt="Download for Windows (x64 ZIP)"/></a> | Extract the ZIP and run the included `.exe`. |
| <a href="https://github.com/marcotag93/TractEdit/releases/download/v3.6.1/TractEdit-3.6.1-macos-arm64.dmg"><img src="https://img.shields.io/badge/Download_for_macOS-DMG_(Apple_Silicon)-30363d?style=for-the-badge&labelColor=24292f&logo=apple&logoColor=white" alt="Download for macOS (Apple Silicon DMG)"/></a> | Open the `.dmg` and install the app. |
| <a href="https://github.com/marcotag93/TractEdit/releases/download/v3.6.1/TractEdit-3.6.1-x86_64.AppImage"><img src="https://img.shields.io/badge/Download_for_Linux-AppImage_(x86__64)-30363d?style=for-the-badge&labelColor=24292f&logo=linux&logoColor=white" alt="Download for Linux (AppImage x86_64)"/></a> | Make the `.AppImage` executable with `chmod +x <filename>.AppImage`, then run it. |
| <a href="https://github.com/marcotag93/TractEdit/releases/download/v3.6.1/TractEdit_3.6.1_amd64.deb"><img src="https://img.shields.io/badge/Download_for_Linux-DEB_(amd64)-30363d?style=for-the-badge&labelColor=24292f&logo=debian&logoColor=white" alt="Download for Linux (DEB amd64)"/></a> | On Debian / Ubuntu, install the `.deb` with `sudo dpkg -i ./TractEdit_3.6.1_amd64.deb`. |

**Windows first launch (optional).** Microsoft Defender may delay the first
launch while scanning bundled libraries. To exclude the TractEdit folder from
scans, run the following command in **PowerShell as administrator**. Replace
the example path with the extracted folder containing `TractEdit.exe`.

```powershell
Add-MpPreference -ExclusionPath 'C:\path\to\TractEdit-3.6.1-windows-x64'
```

The exclusion persists across restarts and reduces antivirus protection for
the folder and its subfolders.

#### Open files from your file manager

Choose **Open with → TractEdit**. For double-click opening, set TractEdit
as the default app for the file type.

Tractograms (`.trk`, `.trx`, `.tck`, `.vtk`, `.vtp`), NIfTI images
(`.nii`, `.nii.gz`) and saved sessions (`.tractedit-session`) open in a new
window. For OBJ meshes (`.obj`), choose RAS or LPS when prompted.

**File associations by platform**

- **Windows.** Choose **Open with → Choose another app** and browse to
  `TractEdit.exe` in the extracted folder. For `.nii.gz`, choose **Just once**
  to keep your usual app for other `.gz` archives.
- **macOS.** Open the DMG and copy `TractEdit.app` to **Applications**. If it
  is missing from Finder's **Open with** menu, choose **Other…** and select
  `TractEdit.app` under **All Applications**. To set the default for all
  `.nii` files, use **Get Info → Open with → Change All**. If Finder groups
  `.nii.gz` with other gzip files, choose **Always Open With** in the app
  picker to set TractEdit for that file only.
- **Linux.** Debian installs the desktop entry. For AppImage integration,
  accept the non-modal prompt on first launch or use **Help → Add TractEdit to
  'Open With'...** (also available to update the path after moving the AppImage
  or to remove the entry). From a terminal you can also run
  `./TractEdit.AppImage --register-file-types` using the actual filename.
  AppImage integration requires `shared-mime-info` and `desktop-file-utils`.

Update checks run automatically by default. Use **Help** to check manually, open the release page when a newer version is available, or disable automatic checks; your preference is saved. Installation remains manual.

<details>
<summary><b>Install from source (Python 3.11 only)</b></summary>

Use Python 3.11 and a compatible C compiler. Installation builds the numerical extension automatically and installs the dependencies listed in [pyproject.toml](pyproject.toml).

```bash
git clone https://github.com/marcotag93/TractEdit.git
cd TractEdit
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in Windows PowerShell, then run:

```bash
python -m pip install .
tractedit
```

If Qt reports a platform error on Linux/Wayland, try `QT_QPA_PLATFORM=xcb tractedit`.

</details>

---

## Keyboard Shortcuts

| Key / Combo                 | Action                                                               |
| --------------------------- | -------------------------------------------------------------------- |
| **s**                 | Add streamlines at cursor to selection (selection grows only)        |
| **Shift+s**           | Remove streamlines at cursor from selection (selection shrinks only) |
| **i**                 | Invert selection                                                     |
| **d**                 | Delete selected streamlines                                          |
| **c**                 | Clear current selection                                              |
| **+ / =**             | Increase selection sphere radius                                     |
| **-**                 | Decrease selection sphere radius                                     |
| **↑ / ↓**           | Axial Slice navigation (Z-axis)                                      |
| **← / →**           | Sagittal Slice navigation (X-axis)                                   |
| **Shift+Scroll**      | Slice navigation on the 2D panel under cursor                        |
| **1**                 | Toggle Pencil drawing mode                                           |
| **2**                 | Toggle Eraser drawing mode                                           |
| **3**                 | Toggle Sphere ROI drawing mode                                       |
| **4**                 | Toggle Rectangle ROI drawing mode                                    |
| **Ctrl+↑ / Ctrl+↓** | Coronal Slice navigation (Y-axis)                                    |
| **Ctrl+Click / Ctrl+Drag** | Move sphere/rectangle ROI (when in mode)                        |
| **Ctrl+Scroll**       | Resize sphere ROI (when in sphere mode)                              |
| **Ctrl+s**            | Save As                                                              |
| **Ctrl+z**            | Undo last streamline, ROI, polyline or surface polygon operation     |
| **Ctrl+y / Ctrl+Shift+z** | Redo last undone streamline, ROI, polyline or surface polygon operation |
| **Ctrl+p**            | Save a screenshot                                                    |
| **Esc**               | Hide selection sphere / cancel active polyline or polygon drawing    |
| **Ctrl+q**            | Quit application                                                     |

---

## Sample Workflow

### Step 1: Load Your Data

```text
File → Load Tractogram... → Load .trk, .tck, .trx, .vtk, or .vtp
File → Load Image...      → Add anatomical image (NIfTI)
```

Example inputs are available in [sample_data/](sample_data/). Replacing a tractogram keeps loaded images, meshes, annotations, ROIs, parcellation and ODF data, while clearing fibre selections and filters.

For **4D images**, right-click the image in the Data Panel and choose **Select volume...**. Three-component images also offer **Display as RGB**. Changing the displayed volume preserves views and ROI edits; ROI masks remain 3D.

For **TRX files**, expand **Data arrays** or **Groups** under the tractogram to inspect embedded values or colour streamlines by component.

<details>
<summary><b>TRX arrays and groups</b></summary>

Expand **Information**, **Data arrays** or **Groups** under the tractogram for per-vertex (DPV), per-streamline (DPS) and per-group (DPG) data. Double-click an array or choose **Inspect values...** for a read-only table with scope selection and on-demand statistics, or select a component to colour streamlines. Streamline and vertex indices are 0-based; components are numbered from 1. Visible scopes include streamlines omitted by Skip. Array names are preserved without inferring anatomical meaning. See the [TRX specification](https://tee-ar-ex.github.io/trx-spec/).

</details>

### Step 2: Navigate

- **2D Slices:** Use arrow keys or Shift+scroll over a panel
- **3D View:** Rotate, zoom and pan with the mouse

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

File → Load ROI...  
Right-click for  
Include/Exclude  

</td>
<td width="25%" align="center">

**✏️ Draw ROIs**

File → New ROI  
`1` Pencil · `2` Eraser  
`3` Sphere · `4` Rectangle  
Ctrl+drag to move  
Include/Exclude  

</td>
<td width="25%" align="center">

**🧠 Parcellation**

File → Load Parcellation...  
Right-click regions  
Include/Exclude  

</td>
</tr>
</table>

*💡 Tip: Use `Ctrl+Z` / `Ctrl+Y` to undo or redo streamline deletions, ROI edits, polyline and surface polygon edits.*

### Step 4: Finalize & Export

| Action | Menu |
| --- | --- |
| Save the edited bundle | File → Save As... |
| Resume work later | File → Save Session |
| Calculate centroid or medoid | File → Calculate Centroid / Calculate Medoid |
| Compute connectivity matrix | File → Compute Connectivity Matrix... |
| Save a density map | File → Save Density Map... |
| Export an offline view | File → Export to HTML... |
| Save a screenshot | Ctrl+P |

Experimental HTML export saves streamlines, slices and ROIs to a simplified offline view for sharing (excluding meshes, annotations, mapping results, parcellations and ODFs).

---

## Surface Meshes

Use **File → Load Mesh...** for `.ply`, `.vtp` or `.obj` files, even without a tractogram or anatomy. Each file becomes one layer under **Meshes**. Select it to control the surface, texture, wireframe, vertices and advanced overlays in the properties panel. Expand the layer to browse child meshes, polylines and surface polygons. Right-click for colour, fit and removal options.

Data file pickers share the last directory you selected, including after a restart. Ctrl-click (Command-click on macOS) to select multiple meshes, ROIs or polyline files in their respective pickers. Meshes load one at a time under the parent selected when you opened the picker. Cancelling a mesh import stops the remaining files and keeps the meshes already loaded.

**Coordinates must already match the subject's physical space in millimetres.** Choose RAS or LPS when importing; OBJ requires an explicit choice. TractEdit does not register meshes or convert voxel or FreeSurfer surface-RAS coordinates.

<details>
<summary><b>Organize meshes and display textures</b></summary>

Select a mesh before loading another to add a child, or select **Meshes** to add a root. Drag layers to reorganize them. Drop a child mesh in the far-left margin of the **Meshes** tree or choose **Move to top level** to make it a root again. This changes organization only. Each mesh retains independent visibility and settings: hiding a parent does not hide its children, while removing a branch removes its descendants and associated annotations.

Enable **Appearance → Surface → Texture** for OBJ meshes. Missing references use compatible, unambiguous same-name assets beside the mesh. **File → Load/Replace Texture...** and the **Texture** context menu apply a chosen PNG/JPEG immediately, with material selection when needed. **Reload Texture...** rereads all associated images; use it when several images have changed on disk. Both commands accept valid MTL updates to colours and UV parameters while retaining existing image associations. An unavailable associated image or invalid material aborts the update and preserves the previous texture.

</details>

<details>
<summary><b>Experimental tractography mapping</b></summary>

With a tractogram loaded, select a mesh and expand **Tractography mapping** in its properties:

- **Endpoint proximity:** find the nearest surface triangle within a chosen distance of each streamline endpoint.
- **Surface intersections:** identify crossings, touches and ambiguous or coplanar contacts along streamline segments.

Choose visible or selected-visible streamlines, then **Compute/Recompute mapping**. Analysis uses the full requested scope regardless of render skip. Review **Diagnostics (QC)** and unassigned or ambiguous endpoints: geometric proximity does not establish an anatomical termination.

Right-click the mapping branch to export complete scientific data (**VTP + CSV + JSON**) or a figure (**PNG + JSON**). Figures reflect display sampling; scientific exports retain complete results. Changed inputs require recomputation. Sessions retain mapping settings and scope, but results must be recomputed after reopening.

</details>

---

## Polylines

Right-click a visible mesh in the 3D view or the Data Panel and choose **Draw Polyline...**. Left-click to place points. Right-click, **Enter** or **Space** saves the trace and exits. During drawing, **Undo** and **Redo** step through picked points. **Backspace** removes the last point and **Esc** discards the draft. Use **Alt+left drag** to rotate while drawing. Segments are straight connections between picked points.

Use **File → Import Polyline...** for standalone CSV or VTP traces, or import from a mesh's context menu to associate them with that mesh. **Load Mesh...** opens a `.vtp` as a single mesh layer; **Import Polyline...** imports each VTP line cell as an editable trace in connectivity order. Choose **From file** to read its SPACE declaration or specify RAS/LPS in millimetres. Select a trace to adjust line width, point size and point visibility. Right-click it to rename, change colour, export or remove it.

Creation, import, rename and removal support **Undo** and **Redo** during the editing session. A multi-line import is one undo step. Saved sessions retain polyline geometry, names and presentation. Polyline undo history lasts until the session is closed.

<details>
<summary><b>Import details and export formats</b></summary>

Each CSV contains one ordered trace: `x,y,z`, optionally preceded by an `index` column. Choose the delimiter, layout and RAS/LPS millimetres in the import dialog. Coordinates must already be aligned; association with a mesh does not project them onto its surface.

Enable **Initial point count** only when the file starts with a point count. For tables with trailing attributes, enable **Ignore columns after XYZ**; those attributes are discarded. Vertex order is preserved without sorting or resampling.

Newly drawn traces and imported traces without a stored colour receive an available colour from the scene; imported colours take precedence. In a mixed CSV/VTP selection, the coordinate choice and layout apply to the CSV files and each VTP uses its own SPACE declaration. Import a VTP separately to specify its space, or select only VTP files to apply one coordinate choice to them all. All traces from one selection are added together after validation.

Export supports `.csv`, `.vtp`, `.vtk`, `.obj`, `.ply`, `.trk` and `.trx`. Choose **CSV XYZ** for coordinates only or **CSV TractEdit** to retain name, colour and display settings. TRK/TRX exports contain geometry only, stored as float32 coordinates. OBJ line colours and PLY edges depend on viewer support.

</details>

---

## Surface Polygons

Right-click a visible triangular OBJ, PLY or VTP mesh in the 3D view or the Data Panel and choose **Draw Surface Polygon...**.
Click along the boundary and use **Alt+drag** to rotate. Each segment follows
mesh edges. Use **Backspace** or **Ctrl+Z** to step back (and **Ctrl+Y** to redo), then **Enter** to close the
boundary. Click a triangle on the side to keep, inspect the coloured area and
press **Enter** again to save. **Esc** discards the draft.

The saved polygon appears under the mesh's **Surface polygons** group. Right-click it to rename, change colour, export or remove it. Choose **Export
Surface Polygon...** to save the closed boundary and original triangles as VTP (`*_polygon.vtp` / `*_patch.vtp`, preserving `original_face_id` and per-corner UVs) or Wavefront OBJ (`*_polygon.obj` / `*_patch.obj`). Use **File → Import Surface Polygon...** or **Import Surface Polygon...** on the source mesh to
import a paired VTP or OBJ boundary and patch (selecting either both files or one file when its sibling is in the same folder).

---

## Save and Restore Sessions

Use **File → Save Session** and **Open Session...** to save and restore edits, ROIs, selections, filters, undo/redo history, views, image volume/RGB mode, TRX component colouring, mesh presentation, polylines and surface polygons in a `.tractedit-session` file. Each save creates a new timestamped archive.

**Keep source files unchanged.** Sessions reference the original tractogram, anatomy, parcellation, ODF, meshes and configured texture assets, including hidden ones. Moved files can be relocated, but missing or changed sources require identical replacements. Polyline and surface polygon data are embedded, so original CSV, VTP and OBJ annotation files are not required.

Finish or cancel active polyline or surface polygon drawing before saving or opening a session. Sessions using 3.6.0 features (such as meshes, polylines, surface polygons or TRX component colouring) cannot be opened in older TractEdit versions.

---

## Command Line

Launch the GUI with files or run conversion and TDI export without opening a window (using `tractedit` from a source or `.deb` installation, or the packaged executable/AppImage):

```bash
tractedit bundle.trk --anat T1w.nii.gz
tractedit bundle.trk --anat T1w.nii.gz --load-roi roi.nii.gz
tractedit bundle.trk --anat T1w.nii.gz --roi 10 20 30 --radius 5
tractedit session.tractedit-session
tractedit input.trk --convert-to output.trx
tractedit input.trk --density-map output.nii.gz --anat T1w.nii.gz
tractedit --help
```

The positional path also accepts `.nii`/`.nii.gz` images and `.obj` meshes. The sphere example uses RAS coordinates and a radius in millimetres.

---

## Contributing and Support

Report bugs or request features through [GitHub Issues](https://github.com/marcotag93/TractEdit/issues). For bugs, include the application version, operating system and steps to reproduce.

<details>
<summary><b>Development setup and checks</b></summary>

From the repository directory, using Python 3.11 and Poetry:

```bash
poetry install --with build,dev,test
poetry run tractedit
poetry run pytest tests/ -v
```

Installation compiles the numerical extension automatically. After changing an AOT kernel in `tractedit_pkg/_numba_aot/kernels/`, rebuild it with:

```bash
poetry run python tractedit_pkg/_numba_aot/build_aot.py
```

[GitHub Actions](.github/workflows/build-releases.yml) defines packaging, scientific and GUI smoke checks for Windows ZIP, macOS DMG, DEB and AppImage, with additional AppImage compatibility checks on Ubuntu 22.04/24.04.

</details>

---

## License

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
