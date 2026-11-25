# -*- coding: utf-8 -*-

"""
Tractedit GUI - Main Application Runner
"""

# ============================================================================
# Imports
# ============================================================================

import os
import sys
import multiprocessing
import pathlib

# PyInstaller freeze support
if __name__ == "__main__":
    multiprocessing.freeze_support()

# ============================================================================
# BLAS Thread Configuration
# ============================================================================
# Prevent BLAS/LAPACK libraries from spawning their own threads, which can
# conflict with the AOT ThreadPoolExecutor-based parallelism used by
# TractEdit's compiled kernels.
if sys.platform == "darwin":  # macOS
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import logging
import importlib.resources
import ctypes
from tractedit_pkg.headless import _run_headless_conversion, _run_headless_density_map
from PyQt6.QtWidgets import QApplication, QSplashScreen
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor, QFont
from PyQt6.QtCore import Qt, QRect, QTimer

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ============================================================================
# Splash Screen
# ============================================================================


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
        self.bar_color = QColor(135, 206, 250)  # progress bar color
        self.text_color = QColor(135, 206, 250)  # progress bar text
        self.setCursor(Qt.CursorShape.WaitCursor)

    def mousePressEvent(self, event):
        """Ignore mouse clicks to prevent splash from closing."""
        event.ignore()

    def keyPressEvent(self, event):
        """Ignore keyboard input to prevent splash from closing."""
        event.ignore()

    def set_progress(self, value, message=None):
        self.progress = value
        if message:
            self.message = message
        self.repaint()  # Force a redraw
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
            self.progress_height,
        )

        text_rect = QRect(
            0, rect.height() - text_space_height, rect.width(), text_space_height
        )

        # Draw Progress Bar
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self.bar_color)
        painter.drawRect(bar_rect)

        # Draw Loading Text
        painter.setPen(self.text_color)
        painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, self.message)




# ============================================================================
# Main Entry Point
# ============================================================================


def main() -> None:
    """
    Main function to start the tractedit application.


    """
    # Application version
    try:
        from tractedit_pkg import __version__ as _version
    except Exception:
        _version = "unknown"

    # Handle version flags before argparse to ensure proper multiline output
    options = sys.argv[1:sys.argv.index("--")] if "--" in sys.argv else sys.argv[1:]
    if sys.platform.startswith("linux") and sys.argv[1:] == ["--register-file-types"]:
        from tractedit_pkg.linux_appimage_integration import run_cli

        sys.exit(run_cli([]))
    if any(flag in options for flag in ("--version", "-V", "-v")):
        _sep = "=" * 60
        print(_sep)
        print("TractEdit ")
        print(f"Version: {_version}")
        print("Author: Marco Tagliaferri, PhD Candidate ")
        print("Center for Mind/Brain Sciences (CIMeC), University of Trento, Italy")
        print("https://github.com/marcotag93/TractEdit")
        print(_sep)
        sys.exit(0)

    from tractedit_pkg import SESSION_EXTENSION
    from tractedit_pkg.cli import build_argument_parser, gui_document_paths

    parser = build_argument_parser()
    args = parser.parse_args()
    if args.bundle and args.bundle.lower().endswith(SESSION_EXTENSION) and (
        args.anat or args.roi_paths or args.roi or args.radius
        or args.convert_to or args.density_map
    ):
        parser.error("Open a saved session on its own, without additional inputs or exports.")

    # Handle headless conversion mode
    if args.convert_to:
        _run_headless_conversion(args.bundle, args.convert_to)
        sys.exit(0)

    # Handle headless density map export
    if args.density_map:
        _run_headless_density_map(args.bundle, args.density_map, args.anat)
        sys.exit(0)

    try:
        args.bundle, args.anat = gui_document_paths(args.bundle, args.anat)
    except ValueError as error:
        parser.error(str(error))

    # Windows: Set app ID for taskbar grouping
    myappid = "tractedit.app.gui"
    if sys.platform == "win32":
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        except Exception as e:
            logger.warning(f"Warning: Could not set AppUserModelID: {e}")

    # Linux: Set argv[0] for X11 WM_CLASS matching
    if sys.platform.startswith("linux"):
        if sys.argv:
            sys.argv[0] = "tractedit"

    if sys.platform == "darwin":
        from tractedit_pkg.macos_documents import DocumentApplication

        app = DocumentApplication(sys.argv)
    else:
        app = QApplication(sys.argv)

    # Linux: Set app identifiers for taskbar icon matching
    if sys.platform.startswith("linux"):
        app.setDesktopFileName("tractedit")
        app.setApplicationName("tractedit")
        app.setApplicationDisplayName("TractEdit")

    # Splash screen and app icon
    splash = None
    try:
        logo_ref = importlib.resources.files("tractedit_pkg.assets").joinpath(
            "logo.png"
        )
        with importlib.resources.as_file(logo_ref) as logo_path:
            if logo_path.is_file():
                pixmap = QPixmap(str(logo_path))
                pixmap = pixmap.scaled(
                    400,
                    400,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                splash = LoadingSplash(pixmap)
                splash.show()

        # Load app icon for window/taskbar
        icon_ref = importlib.resources.files("tractedit_pkg.assets").joinpath(
            "tractedit.png"
        )
        with importlib.resources.as_file(icon_ref) as icon_path:
            if icon_path.is_file():
                app_icon = QIcon()
                icon_pixmap = QPixmap(str(icon_path))
                for size in [16, 24, 32, 48, 64, 128, 256]:
                    scaled = icon_pixmap.scaled(
                        size,
                        size,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                    app_icon.addPixmap(scaled)
                app.setWindowIcon(app_icon)
                logger.info(f"Loaded app icon from: {icon_path}")

    except Exception as e:
        logger.warning(f"Could not load splash screen assets: {e}")

    # ------------------------------------------------------------------
    # Parallel Background Import (VTK+FURY / Nibabel+SciPy)
    # ------------------------------------------------------------------
    #   Lane 0 — VTK then FURY: FURY imports VTK internally, so they are
    #             serialized by Python's per-module import lock regardless,
    #             sharing a single thread
    #   Lane 1 — Nibabel + TRX + SciPy: independent, running them sequentially in
    #             one thread halves I/O and GIL contention.
    # ------------------------------------------------------------------
    import threading
    import time
    import numpy  # noqa: F401  — pre-warm before threads start

    _lane_events: list[threading.Event] = [threading.Event(), threading.Event()]
    _import_errors: list[Exception | None] = [None, None]

    _LANE_VTK_FURY = 0
    _LANE_NIB_SCIPY = 1

    def _import_vtk_and_fury() -> None:
        """Lane 0: VTK then FURY — sequential by dependency (FURY needs VTK)."""
        try:
            import vtk  # noqa: F811, F401
            from vtk.util import numpy_support  # noqa: F401
            import fury  # noqa: F401
            from fury import actor, window, colormap  # noqa: F401
        except Exception as exc:
            _import_errors[_LANE_VTK_FURY] = exc
        finally:
            _lane_events[_LANE_VTK_FURY].set()

    def _import_nibabel_and_scipy() -> None:
        """Lane 1: Nibabel, TRX, and SciPy — all independent of VTK/FURY."""
        try:
            import nibabel  # noqa: F401
            import nibabel.streamlines  # noqa: F401
            import nibabel.streamlines.array_sequence  # noqa: F401
            import nibabel.streamlines.tractogram  # noqa: F401
            import nibabel.streamlines.trk  # noqa: F401
            import nibabel.streamlines.tck  # noqa: F401
            import trx.trx_file_memmap  # noqa: F401
            from scipy.ndimage import gaussian_filter  # noqa: F401
            from scipy.ndimage import binary_dilation  # noqa: F401
            from scipy.special import sph_harm_y  # noqa: F401
        except Exception as exc:
            _import_errors[_LANE_NIB_SCIPY] = exc
        finally:
            _lane_events[_LANE_NIB_SCIPY].set()

    for _target in [_import_vtk_and_fury, _import_nibabel_and_scipy]:
        threading.Thread(target=_target, daemon=True).start()

    if splash:
        splash.set_progress(5, "Loading VTK / FURY / Nibabel...")

    # Wait for Lane 1 (faster) while keeping the splash responsive
    while not _lane_events[_LANE_NIB_SCIPY].is_set():
        if splash:
            QApplication.processEvents()
        time.sleep(0.05)

    if splash:
        splash.set_progress(20, "Loading VTK / FURY...")

    # Wait for Lane 0 (VTK + FURY — the long pole)
    while not _lane_events[_LANE_VTK_FURY].is_set():
        if splash:
            QApplication.processEvents()
        time.sleep(0.05)

    if splash:
        splash.set_progress(40, "Finalizing imports...")

    logger.info("Background library imports complete.")

    # QVTKRenderWindowInteractor needs both VTK and Qt
    try:
        from vtkmodules.qt.QVTKRenderWindowInteractor import (  # noqa: F401
            QVTKRenderWindowInteractor,
        )
    except Exception as exc:
        logger.warning("QVTKRenderWindowInteractor import failed: %s", exc)

    # Report any lane errors (non-fatal).
    _lane_labels = ["VTK+FURY", "Nibabel+SciPy"]
    for idx, err in enumerate(_import_errors):
        if err is not None:
            logger.warning(
                "%s background import failed: %s", _lane_labels[idx], err
            )

    # Configure VTK on the main thread
    if _import_errors[_LANE_VTK_FURY] is None:
        try:
            import vtk 

            out_window = vtk.vtkOutputWindow()
            vtk.vtkOutputWindow.SetInstance(out_window)
            vtk.vtkObject.GlobalWarningDisplayOff()
            logger.info("VTK initialized successfully.")
        except Exception as e:
            logger.warning("Error suppressing VTK output: %s", e)
    else:
        logger.warning("VTK import failed: %s", _import_errors[_LANE_VTK_FURY])

    # Main Window import
    if splash:
        splash.set_progress(50, "Loading application modules...")
    logger.info("Loading main window module...")

    try:
        from tractedit_pkg.main_window import MainWindow

        logger.info("Main window module loaded.")
    except ImportError as e:
        logger.error(f"Error importing necessary modules: {e}")
        sys.exit(1)

    # Main Window
    if splash:
        splash.set_progress(75, "Building User Interface...")
    logger.info("Building main window UI...")

    main_window: MainWindow = MainWindow()
    logger.info("Main window created.")

    if sys.platform == "darwin" and not (
        args.bundle or args.anat or args.roi_paths or args.roi
    ):
        app.reserve_initial_window(main_window)

    if splash:
        splash.set_progress(90, "Starting...")

    # Launch - start maximized
    logger.info("Launching application window...")
    main_window.showMaximized()
    main_window.update_notifier.start()
    main_window.appimage_integration.start()

    if splash:
        splash.finish(main_window)

    # Trigger initial file loading if arguments provided
    if sys.platform == "darwin":
        app.finish_startup(args.bundle, args.anat)
    if args.bundle or args.anat or args.roi_paths or args.roi:
        # QTimer to allow the event loop to start first
        QTimer.singleShot(
            100,
            lambda: main_window.load_initial_files(
                args.bundle, args.anat, args.roi_paths, args.roi, args.radius
            ),
        )

    logger.info("App started.")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
