# TractEdit

A Python-based graphical interface for interactively **viewing**, **selecting**, and **editing** tractography bundles in `.trk` and `.tck` formats.




https://github.com/user-attachments/assets/7558b0ab-3fc0-44d3-9eb6-671b29608f00


---

## Author

**Marco Tagliaferri**, PhD Candidate in Neuroscience  
Center for Mind/Brain Sciences (CIMeC), University of Trento Italy 
📧 marco.tagliaferri@unitn.it / marco.tagliaferri93@gmail.com

---

## Key Features

- **Load & Save** streamline bundles (`.trk`, `.tck`)
- **3D Visualization** with [VTK](https://vtk.org/) and [FURY](https://fury.gl/)
    - Default gray, orientation (RGB), or scalar-based coloring
- **Interactive Editing Tools:**
    - Sphere-based streamline selection (with adjustable radius)
    - Streamline deletion and undo/redo support
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

- Python 3.8–3.11 (tested)
- [PyQt6](https://pypi.org/project/PyQt6/)
- [VTK](https://vtk.org/)
- [FURY](https://fury.gl/)
- [Nibabel](https://nipy.org/nibabel/)
- NumPy
- pytz
  
Recommend a virtual environment:
```bash
python -m venv venv
source venv/bin/activate 
pip install -r requirements.txt
```

### 3. Launch the App
```bash
python3 main.py
```

> On some Linux systems (e.g., Ubuntu Wayland), if you encounter Qt errors, try 'xcb' below. The bash command automatically handles this error.
```bash
QT_QPA_PLATFORM=xcb python3 main.py
```

### 4. Load Sample Data (Optional)
Explore `sample_data/` to test TractEdit with example `.trk` or `.tck` files.

---

### For Linux/MacOS
```bash
chmod +x Tractedit
./Tractedit
```
You can add TractEdit to your PATH:
```bash
echo 'export PATH="/path/to/tractedit:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

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
| **Ctrl+s**           | Save As                                     |
| **Ctrl+z**           | Undo last deletion                          |
| **Ctrl+y / Shift+z** | Redo last undone deletion                   |
| **Ctrl+p**           | Save a screenshot                           |
| **Esc**              | Hide selection sphere                       |
| **Ctrl+q**           | Quit application                            |

---

## Sample Workflow

1. Open a `.trk` or `.tck` file via **File → Open**
2. Use the mouse and `S` key to select streamlines
3. Press `D` to delete, or `C` to clear selection
4. Change streamline color in **View → Streamline Color**
5. Save your edited bundle with **File → Save As**

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
