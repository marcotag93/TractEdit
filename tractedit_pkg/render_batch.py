"""Synchronous render batching for compound GUI actions."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps


_active = ContextVar("tractedit_render_batch", default=None)


def request_render(panel, window):
    batch = _active.get()
    if batch is not None and batch[0] is panel:
        batch[1][window] = None
    else:
        window.Render()


@contextmanager
def defer_render(panel):
    """Publish each requested window once when the outer action succeeds."""
    batch = _active.get()
    if batch is not None and batch[0] is panel:
        yield
        return
    windows = {}
    token = _active.set((panel, windows))
    try:
        yield
    finally:
        _active.reset(token)
    for window in windows:
        window.Render()


def batch_render(panel_for):
    def decorate(action):
        @wraps(action)
        def run(owner, *args, **kwargs):
            with defer_render(panel_for(owner)):
                return action(owner, *args, **kwargs)

        return run

    return decorate
