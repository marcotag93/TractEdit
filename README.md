# TractEdit

A Python-based graphical interface for interactively **viewing**, **selecting**, and **editing** tractography bundles in `.trk`, `.tck` and `.trx` formats.









https://github.com/user-attachments/assets/a6543a1e-fc86-4450-a827-fcee61b205be





---

## Author

**Marco Tagliaferri**, PhD Candidate in Neuroscience  
Center for Mind/Brain Sciences (CIMeC), University of Trento Italy 

📧 marco.tagliaferri@unitn.it / marco.tagliaferri93@gmail.com

---

## Key Features

- **Load & Save** streamline bundles (`.trk`, `.tck`, `.trx`)
- **Multi-View Orthogonal Visualization:** Integrated 3D viewer and three linked 2D orthogonal slice views (Axial, Coronal, Sagittal).
- **Anatomical Image:** Load NIfTI images (`.nii`, `.nii.gz`) as backgrounds for anatomical context and **interactive slice navigation**.
- **3D Visualization** with [VTK](https://vtk.org/) and [FURY](https://fury.gl/)
    - Default orientation (RGB), or scalar-based coloring with dynamic colormap range adjustment, or greyscale.
- **Interactive Editing Tools:**
    - Sphere-based streamline selection (with adjustable radius)
    - Streamline deletion and undo/redo support.
    - Screenshot export
- **Streamline Info Display:**
    - File name, streamline count, voxel size, bounding box, etc.
- **Keyboard Shortcuts** for fast interaction (see full list below)

> ⚠️ **Note:** TractEdit is optimized for refining *bundles*, not whole-brain tractograms. Files with a large number of streamlines may cause slowdowns or freezing depending on your system.
---

## Getting Started (Manual Install)

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/tractedit.git
cd tractedit
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
  
Recommend a virtual environment:
```bash
python -m venv venv
source venv/bin/activate 
# Install the project and its dependencies:
pip install .
```

### 3. Launch the App
The application can now be launched using the tractedit command installed via pip.
```bash
tractedit
```

> On some Linux systems (e.g., Ubuntu Wayland), if you encounter Qt errors, try 'xcb' below.
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

1. Open a `.trk`, `.tck` or `.trx` file via **File → Open**
2. Load an anatomical image via **File → Load Image** to enable 2D slice views.
3. Use the mouse click-drag in the 2D slice views or the arrow keys (see shortcuts above) to navigate the anatomical slices.
4. Use the mouse and `S` key to select streamlines
5. Press `D` to delete, or `C` to clear selection. Use Ctrl+Z to undo deletions.
6. If needed, change streamline color in **View → Streamline Color**. If using Color by Scalar, use the Scalar Range toolbar at the top of the window to adjust the min/max range of the colormap.
7. Save your edited bundle with **File → Save As**

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
