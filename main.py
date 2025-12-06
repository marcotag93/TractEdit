# -*- coding: utf-8 -*-

"""
Tractedit GUI - Main Application Runner
"""

import os
import sys
import logging
import importlib.resources 

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', 
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

try:
    import vtk
    out_window: vtk.vtkOutputWindow = vtk.vtkOutputWindow()
    vtk.vtkOutputWindow.SetInstance(out_window)
    vtk.vtkObject.GlobalWarningDisplayOff()
except ImportError:
    logger.warning("Warning: VTK not found, cannot suppress output.")
except Exception as e:
    logger.warning(f"Warning: Error suppressing VTK output: {e}")

try:
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtGui import QIcon
    from tractedit_pkg.main_window import MainWindow
except ImportError as e:
    logger.error(f"Error importing necessary modules: {e}")
    logger.error("Please ensure you have installed the package correctly (e.g., 'pip install .')")
    sys.exit(1)

def main() -> None:
    """
    Main function to start the tractedit application.
    """
    app: QApplication = QApplication(sys.argv)

    try:
        logo_path_obj = importlib.resources.files('tractedit_pkg.assets').joinpath('logo.png')
        
        if logo_path_obj.is_file():
            app.setWindowIcon(QIcon(str(logo_path_obj)))
        else:
            logger.warning(f"Warning: Application icon file not found in package assets.")
    except Exception as e:
            logger.warning(f"Warning: Could not load application icon: {e}")

    main_window: MainWindow = MainWindow()
    main_window.show()

    logger.info("Starting app...")
    sys.exit(app.exec())

if __name__ == "__main__":
    main()