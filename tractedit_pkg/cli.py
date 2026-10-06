# -*- coding: utf-8 -*-

"""Command-line argument definitions for TractEdit."""

import argparse


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the parser shared by GUI and headless entry points."""
    parser = argparse.ArgumentParser(description="TractEdit - GUI")
    parser.add_argument(
        "bundle",
        nargs="?",
        help="Path to the bundle file (.trk, .tck, .trx, .vtk, .vtp)",
    )
    parser.add_argument("--anat", help="Path to the anatomical image (T1w)")
    parser.add_argument(
        "--load-roi",
        action="append",
        dest="roi_paths",
        help="Path to ROI image(s). Can be used multiple times.",
    )
    parser.add_argument(
        "--roi",
        nargs=3,
        type=float,
        action="append",
        metavar=("X", "Y", "Z"),
        help=(
            "Create a sphere ROI at these coordinates (X Y Z). "
            "Can be used multiple times."
        ),
    )
    parser.add_argument(
        "--radius",
        type=float,
        action="append",
        help="Radius of the sphere ROI (default: 5mm). Can be used multiple times.",
    )
    parser.add_argument(
        "--convert-to",
        dest="convert_to",
        metavar="OUTPUT",
        help=(
            "Headless conversion: convert bundle to OUTPUT file format without GUI. "
            "Supported formats: .trk, .tck, .trx, .vtk, .vtp"
        ),
    )
    parser.add_argument(
        "--density-map",
        dest="density_map",
        metavar="OUTPUT",
        help=(
            "Headless export: compute and save density map as NIfTI (.nii.gz) "
            "without GUI. Use --anat to align to anatomical image grid."
        ),
    )
    return parser
