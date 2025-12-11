# TractEdit

A Python-based graphical interface for interactively **viewing**, **selecting**, and **editing** tractography bundles in `.trk`, `.tck` and `.trx` formats.









https://github.com/user-attachments/assets/845e0c0e-8bf4-430b-b660-01adeba8a4b9







---

## Author

**Marco Tagliaferri**, PhD Candidate in Neuroscience  
Center for Mind/Brain Sciences (CIMeC), University of Trento Italy 

📧 marco.tagliaferri@unitn.it / marco.tagliaferri93@gmail.com

---

## Key Features

- **Load & Save** streamline bundles (`.trk`, `.tck`, `.trx`)
- **Whole-Brain Tractogram Support:** Optimized for large datasets (>2 million streamlines) using **asynchronous file loading**, **Numba-optimized** geometry calculations, and **vectorized VTK rendering**. Includes a stride-based "skip" feature for smooth interaction. 
- **Multi-View Orthogonal Visualization:** Integrated 3D viewer and three linked 2D orthogonal slice views (Axial, Coronal, Sagittal).
- **Anatomical Image:** Load NIfTI images (`.nii`, `.nii.gz`) as backgrounds for anatomical context and **interactive slice navigation**.
- **Multi-Layer Anatomical ROI Support:** - Load multiple NIfTI images (`.nii`, `.nii.gz`) as background Region of Interest (ROI) layers.
    - Independent visibility toggles and color settings.
    - **Logical Filtering:** Right-click ROIs to set them as exclusion or inclusion ROIs for streamlines.
- **3D Visualization** with [VTK](https://vtk.org/) and [FURY](https://fury.gl/)
    - Default orientation (RGB), or scalar-based coloring with dynamic colormap range adjustment, or greyscale.
    - Interactive RAS coordinate bar for precise navigation.
    - ODF/Glyph Visualization: View Spherical Harmonics (SH) coefficients with a "Tunnel View" that masks glyphs to the bundle's path.
- **Interactive Editing Tools:**
    - Sphere-based streamline selection (with adjustable radius)
    - Streamline deletion and undo/redo support.
    - Screenshot export
- **Bundle Analytics:**
    - Calculate **Centroid** and **Medoid** of the edited bundle.
    - Generate and save **Density Map (TDI)** from the edited bundle.
- **Streamline Info Display:**
    - File name, streamline count, voxel size, bounding box, etc.
    - Vertical data panel with hover details 
- **Keyboard Shortcuts** for fast interaction (see full list below)
- **Fast Startup:** Splash screen implementation for immediate feedback and optimized library loading.

> ⚠️ **Note:** While TractEdit supports **whole-brain tractograms**, rendering density may be automatically reduced (skipped) for extremely large files to maintain interactivity. You can adjust this manually in the toolbar.
>
> **Selection Strategy:**
> * **Sphere Selection:** Recommended for refining specific **bundles** or small bundle complexes. Note that this method is **ineffective** on whole-brain datasets if streamlines are being "skipped" (hidden) for performance, as the sphere can only interact with visible fibers.
> * **ROI Filtering:** Recommended for **whole-brain tractograms**. Loading NIfTI images as logic filters (Include/Exclude) works on the entire dataset regardless of visual density.
> * *(Future Roadmap):* Hand-drawing ROIs directly on slices is planned for a future update to provide an additional layer of flexible filtering.
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
- pytz
- [trx-python](https://pypi.org/project/trx-python/)
- Numba
- Scipy
  
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

> On certain Linux systems (e.g., Ubuntu on Wayland), you may encounter Qt platform errors. If so, try the xcb workaround below.
```bash
QT_QPA_PLATFORM=xcb tractedit
```

### 4. Load Sample Data (Optional)
Explore `sample_data/` to test TractEdit with example `.trk`, `.tck` or `.trx` files.

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
| **Ctrl+↑ / Ctrl+↓**  | Coronal Slice navigation (Y-axis)           |
| **Ctrl+s**           | Save As                                     |
| **Ctrl+z**           | Undo last deletion                          |
| **Ctrl+y / Shift+z** | Redo last undone deletion                   |
| **Ctrl+p**           | Save a screenshot                           |
| **Esc**              | Hide selection sphere                       |
| **Ctrl+q**           | Quit application                            |

---

## Sample Workflow

1. Open a `.trk`, `.tck` or `.trx` file via **File → Open**.
2. Load an anatomical image via **File → Load Image**.
3. Load anatomical ROIs via **File → Load ROI** and right-click ROI layers to set them as Include or Exclude regions to automatically filter streamlines.
4. Use the mouse click-drag in the 2D slice views or the arrow keys (see shortcuts above) to navigate the anatomical slices.
5. Use the mouse and `S` key to select streamlines.
6. Press `D` to delete, or `C` to clear selection. Use **Ctrl+Z** to undo deletions.
7. Customize the view using **View → Streamline Geometry** (Lines/Tubes) or load ODFs via **File → Load ODF**.
8. If needed, change streamline color in **View → Streamline Color**. If using Color by Scalar, use the Scalar Range toolbar. If using ODF, use **View → Show ODF Tunnel**. 
9. Save the centroid and/or the medoid of your edited bundle with **File → Calculate Centroid** and **File → Calculate Medoid**.
10. Generate a density map of your edited bundle via **File → Save Density Map**.
11. Save your edited bundle with **File → Save As**.

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
