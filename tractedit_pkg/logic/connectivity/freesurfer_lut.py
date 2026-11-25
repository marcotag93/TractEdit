"""FreeSurfer label defaults and external colour-table parsing."""

import logging
import os
from typing import Optional, Dict

from .. import connectivity

logger = logging.getLogger(__name__)


# Default FreeSurfer LUT location (can be overridden)
FREESURFER_LUT_PATHS = [
    os.path.join(os.environ.get("FREESURFER_HOME", ""), "FreeSurferColorLUT.txt"),
    os.path.expanduser("~/.freesurfer/FreeSurferColorLUT.txt"),
]

# Embedded FreeSurfer aparc+aseg labels (most common labels). This ensures labels are always available even without FreeSurfer installation
FREESURFER_BUILTIN_LABELS = {
    0: "Unknown",
    # Subcortical structures
    2: "Left-Cerebral-White-Matter",
    3: "Left-Cerebral-Cortex",
    4: "Left-Lateral-Ventricle",
    5: "Left-Inf-Lat-Vent",
    7: "Left-Cerebellum-White-Matter",
    8: "Left-Cerebellum-Cortex",
    10: "Left-Thalamus",
    11: "Left-Caudate",
    12: "Left-Putamen",
    13: "Left-Pallidum",
    14: "3rd-Ventricle",
    15: "4th-Ventricle",
    16: "Brain-Stem",
    17: "Left-Hippocampus",
    18: "Left-Amygdala",
    24: "CSF",
    26: "Left-Accumbens-area",
    28: "Left-VentralDC",
    30: "Left-vessel",
    31: "Left-choroid-plexus",
    41: "Right-Cerebral-White-Matter",
    42: "Right-Cerebral-Cortex",
    43: "Right-Lateral-Ventricle",
    44: "Right-Inf-Lat-Vent",
    46: "Right-Cerebellum-White-Matter",
    47: "Right-Cerebellum-Cortex",
    49: "Right-Thalamus",
    50: "Right-Caudate",
    51: "Right-Putamen",
    52: "Right-Pallidum",
    53: "Right-Hippocampus",
    54: "Right-Amygdala",
    58: "Right-Accumbens-area",
    60: "Right-VentralDC",
    62: "Right-vessel",
    63: "Right-choroid-plexus",
    72: "5th-Ventricle",
    77: "WM-hypointensities",
    78: "Left-WM-hypointensities",
    79: "Right-WM-hypointensities",
    80: "non-WM-hypointensities",
    81: "Left-non-WM-hypointensities",
    82: "Right-non-WM-hypointensities",
    85: "Optic-Chiasm",
    251: "CC_Posterior",
    252: "CC_Mid_Posterior",
    253: "CC_Central",
    254: "CC_Mid_Anterior",
    255: "CC_Anterior",
    # Desikan-Killiany atlas (aparc) - Left hemisphere (1000s)
    1000: "ctx-lh-unknown",
    1001: "ctx-lh-bankssts",
    1002: "ctx-lh-caudalanteriorcingulate",
    1003: "ctx-lh-caudalmiddlefrontal",
    1005: "ctx-lh-cuneus",
    1006: "ctx-lh-entorhinal",
    1007: "ctx-lh-fusiform",
    1008: "ctx-lh-inferiorparietal",
    1009: "ctx-lh-inferiortemporal",
    1010: "ctx-lh-isthmuscingulate",
    1011: "ctx-lh-lateraloccipital",
    1012: "ctx-lh-lateralorbitofrontal",
    1013: "ctx-lh-lingual",
    1014: "ctx-lh-medialorbitofrontal",
    1015: "ctx-lh-middletemporal",
    1016: "ctx-lh-parahippocampal",
    1017: "ctx-lh-paracentral",
    1018: "ctx-lh-parsopercularis",
    1019: "ctx-lh-parsorbitalis",
    1020: "ctx-lh-parstriangularis",
    1021: "ctx-lh-pericalcarine",
    1022: "ctx-lh-postcentral",
    1023: "ctx-lh-posteriorcingulate",
    1024: "ctx-lh-precentral",
    1025: "ctx-lh-precuneus",
    1026: "ctx-lh-rostralanteriorcingulate",
    1027: "ctx-lh-rostralmiddlefrontal",
    1028: "ctx-lh-superiorfrontal",
    1029: "ctx-lh-superiorparietal",
    1030: "ctx-lh-superiortemporal",
    1031: "ctx-lh-supramarginal",
    1032: "ctx-lh-frontalpole",
    1033: "ctx-lh-temporalpole",
    1034: "ctx-lh-transversetemporal",
    1035: "ctx-lh-insula",
    # Desikan-Killiany atlas (aparc) - Right hemisphere (2000s)
    2000: "ctx-rh-unknown",
    2001: "ctx-rh-bankssts",
    2002: "ctx-rh-caudalanteriorcingulate",
    2003: "ctx-rh-caudalmiddlefrontal",
    2005: "ctx-rh-cuneus",
    2006: "ctx-rh-entorhinal",
    2007: "ctx-rh-fusiform",
    2008: "ctx-rh-inferiorparietal",
    2009: "ctx-rh-inferiortemporal",
    2010: "ctx-rh-isthmuscingulate",
    2011: "ctx-rh-lateraloccipital",
    2012: "ctx-rh-lateralorbitofrontal",
    2013: "ctx-rh-lingual",
    2014: "ctx-rh-medialorbitofrontal",
    2015: "ctx-rh-middletemporal",
    2016: "ctx-rh-parahippocampal",
    2017: "ctx-rh-paracentral",
    2018: "ctx-rh-parsopercularis",
    2019: "ctx-rh-parsorbitalis",
    2020: "ctx-rh-parstriangularis",
    2021: "ctx-rh-pericalcarine",
    2022: "ctx-rh-postcentral",
    2023: "ctx-rh-posteriorcingulate",
    2024: "ctx-rh-precentral",
    2025: "ctx-rh-precuneus",
    2026: "ctx-rh-rostralanteriorcingulate",
    2027: "ctx-rh-rostralmiddlefrontal",
    2028: "ctx-rh-superiorfrontal",
    2029: "ctx-rh-superiorparietal",
    2030: "ctx-rh-superiortemporal",
    2031: "ctx-rh-supramarginal",
    2032: "ctx-rh-frontalpole",
    2033: "ctx-rh-temporalpole",
    2034: "ctx-rh-transversetemporal",
    2035: "ctx-rh-insula",
}


def parse_freesurfer_lut(lut_path: Optional[str] = None) -> Dict[int, str]:
    """
    Parses a FreeSurfer color LUT file to extract label-to-name mapping.

    Priority: External LUT file > Built-in labels

    Args:
        lut_path: Path to FreeSurfer LUT file. If None, searches default paths.

    Returns:
        Dictionary mapping label IDs to region names.
    """
    # First, try to find and parse external LUT file (higher priority)
    if lut_path and os.path.isfile(lut_path):
        paths_to_try = [lut_path]
    else:
        paths_to_try = connectivity.FREESURFER_LUT_PATHS

    lut_file = None
    for path in paths_to_try:
        if os.path.isfile(path):
            lut_file = path
            break

    # If external LUT found, parse it
    if lut_file is not None:
        label_names: Dict[int, str] = {0: "Unknown"}
        try:
            with open(lut_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and empty lines
                    if not line or line.startswith("#"):
                        continue

                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            label_id = int(parts[0])
                            label_name = parts[1]
                            label_names[label_id] = label_name
                        except ValueError:
                            continue

            logger.info(
                f"Loaded {len(label_names)} labels from FreeSurfer LUT: {lut_file}"
            )
            return label_names
        except (OSError, ValueError) as e:
            logger.warning(f"Failed to parse FreeSurfer LUT ({lut_file}): {e}")
            # Fall through to use built-in labels

    # Fallback to built-in labels
    logger.info("Using built-in FreeSurfer labels (no external LUT file found).")
    return connectivity.FREESURFER_BUILTIN_LABELS.copy()
