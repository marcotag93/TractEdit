# -*- coding: utf-8 -*-

"""
TractEdit package initialization.
"""

import logging

logger = logging.getLogger(__name__)

#: Application version — keep in sync with ``pyproject.toml``.
__version__: str = "3.4.5"

try:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("tractedit")
except Exception as e:
    logger.debug("Could not read installed package version: %s", e)
