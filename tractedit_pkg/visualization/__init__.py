# -*- coding: utf-8 -*-

"""
Visualization package for TractEdit.

This package provides modular components for VTK/FURY visualization,
including interactor styles, actor factories, scene management, and
the main VTKPanel assembly.
"""

from .vtk_panel import VTKPanel
from .interactor import CustomInteractorStyle2D
from .html_export import export_to_html

__all__ = ["VTKPanel", "CustomInteractorStyle2D", "export_to_html"]
