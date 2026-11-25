"""Shared directory preference for the application's data pickers."""

import os


def choose_input(picker, parent, title, directory, filters):
    settings = getattr(getattr(parent, "mw", parent), "settings", None)
    key = "files/lastOpenDirectory"
    if settings is not None:
        previous = settings.value(key, "")
        if isinstance(previous, str) and os.path.isdir(previous):
            directory = previous
    paths, selected_filter = picker(parent, title, directory, filters)
    if paths and settings is not None:
        first = paths if isinstance(paths, str) else paths[0]
        settings.setValue(key, os.path.dirname(os.path.abspath(first)))
    return paths, selected_filter
