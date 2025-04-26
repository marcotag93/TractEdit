# -*- coding: utf-8 -*-

"""
Tractedit GUI - Main Application Runner
"""

import sys
import os

try:
    import vtk
    out_window = vtk.vtkOutputWindow()
    vtk.vtkOutputWindow.SetInstance(out_window)
    vtk.vtkObject.GlobalWarningDisplayOff()
except ImportError:
    print("Warning: VTK not found, cannot suppress output.")
except Exception as e:
    print(f"Warning: Error suppressing VTK output: {e}")

script_dir = os.path.dirname(__file__)
package_dir = os.path.join(script_dir, 'tractedit_pkg')

if package_dir not in sys.path:
    sys.path.insert(0, script_dir)

try:
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtGui import QIcon
    from tractedit_pkg.main_window import MainWindow
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    sys.exit(1)

def main():
    """
    Main function to start the tractedit application.
    """
    app = QApplication(sys.argv)

    logo_path = os.path.join(os.path.dirname(__file__), 'tractedit_pkg', 'assets', 'logo.png')
    if os.path.exists(logo_path):
        app.setWindowIcon(QIcon(logo_path))
    else:
        print(f"Warning: Application icon file not found at {logo_path}")

    main_window = MainWindow()
    main_window.show()

    print("Starting app...")
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
