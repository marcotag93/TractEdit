# -*- coding: utf-8 -*-

"""
Tractedit GUI - Main Application Runner
"""

import os
import sys
import logging
import importlib.resources
import ctypes
from PyQt6.QtWidgets import QApplication, QSplashScreen
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor, QFont
from PyQt6.QtCore import Qt, QRect 

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', 
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class LoadingSplash(QSplashScreen):
    """
    Custom Splash Screen with a Progress Bar drawn at the bottom.
    """
    def __init__(self, pixmap, flags=Qt.WindowType.WindowStaysOnTopHint):
        super().__init__(pixmap, flags)
        self.progress = 0
        self.message = "Initializing..."
        
        # UI Settings
        self.progress_height = 20
        self.bar_color = QColor(135, 206, 250)    # progress bar color
        self.text_color = QColor(135, 206, 250)   # progress bar text
        self.setCursor(Qt.CursorShape.WaitCursor)

    def set_progress(self, value, message=None):
        self.progress = value
        if message:
            self.message = message
        self.repaint() # Force a redraw
        QApplication.processEvents() 

    def drawContents(self, painter: QPainter):
        
        # Draw the Pixmap (Logo)
        super().drawContents(painter)

        # Setup Geometry 
        rect = self.rect()
        text_space_height = 30
        
        bar_y_pos = rect.height() - self.progress_height - text_space_height
        
        bar_rect = QRect(
            0, 
            bar_y_pos, 
            int(rect.width() * (self.progress / 100)), 
            self.progress_height
        )
        
        text_rect = QRect(
            0, 
            rect.height() - text_space_height,
            rect.width(), 
            text_space_height
        )
        
        # Draw Progress Bar
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self.bar_color)
        painter.drawRect(bar_rect)

        # Draw Loading Text
        painter.setPen(self.text_color)
        painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, self.message)

def main() -> None:
    """
    Main function to start the tractedit application.
    """
    # Windows App ID 
    myappid = 'tractedit.app.gui' 
    if sys.platform == 'win32':
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        except Exception as e:
            logger.warning(f"Warning: Could not set AppUserModelID: {e}")

    # Start Application (Instant)
    app: QApplication = QApplication(sys.argv)

    # Start SplashScreen (Before importing heavy libraries)
    splash = None
    try:
        # Load larger logo for splash if available, or standard logo
        logo_ref = importlib.resources.files('tractedit_pkg.assets').joinpath('logo.png')
        
        with importlib.resources.as_file(logo_ref) as logo_path:
            if logo_path.is_file():
                pixmap = QPixmap(str(logo_path))
                pixmap = pixmap.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                
                splash = LoadingSplash(pixmap)
                splash.show()
                
                # Set App Icon
                app_icon = QIcon(str(logo_path))
                app.setWindowIcon(app_icon)
                
    except Exception as e:
        logger.warning(f"Could not load splash screen assets: {e}")

    # VTK (goes first)
    if splash: splash.set_progress(10, "Initializing VTK System...")
    
    try:
        import vtk
        out_window: vtk.vtkOutputWindow = vtk.vtkOutputWindow()
        vtk.vtkOutputWindow.SetInstance(out_window)
        vtk.vtkObject.GlobalWarningDisplayOff()
    except ImportError:
        logger.warning("Warning: VTK not found.")
    except Exception as e:
        logger.warning(f"Warning: Error suppressing VTK output: {e}")

    # Heavy imports
    if splash: splash.set_progress(30, "Loading Libraries...")
    
    try:
        from tractedit_pkg.main_window import MainWindow
    except ImportError as e:
        logger.error(f"Error importing necessary modules: {e}")
        sys.exit(1)

    # Main Window
    if splash: splash.set_progress(70, "Building User Interface...")

    main_window: MainWindow = MainWindow()
    
    if splash: splash.set_progress(90, "Starting...")

    # Launch 
    main_window.show()
    
    if splash:
        splash.finish(main_window)

    logger.info("App started.")
    sys.exit(app.exec())

if __name__ == "__main__":
    main()