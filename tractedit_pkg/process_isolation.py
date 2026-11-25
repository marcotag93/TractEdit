"""Spawn and retire native workers without importing GUI or rendering modules."""

import multiprocessing
import sys
from contextlib import ExitStack
from pathlib import Path
from threading import Lock

_spawn_lock = Lock()


def start_process(process):
    package_root = str(Path(__file__).resolve().parent.parent)
    with _spawn_lock:
        sys.path.insert(0, package_root)
        try:
            process.start()
        finally:
            sys.path.remove(package_root)


def run_isolated_process(
    target, args, *, cancel_check, progress=None, channel=False, cooperative=False,
    kwargs=None, start=start_process,
):
    """Run a spawn-safe target. cancel_check raises its protocol's cancellation error."""
    context = multiprocessing.get_context("spawn")
    cancelled = context.Event() if cooperative else None
    if cooperative:
        args = (*args, cancelled)
    with ExitStack() as resources:
        receiver = None
        if channel:
            receiver, sender = context.Pipe(duplex=False)
            resources.callback(receiver.close)
            resources.callback(sender.close)
            args = (*args, sender)

        def drain():
            while receiver is not None and not receiver.closed:
                cancel_check()
                try:
                    if not receiver.poll():
                        break
                    message = receiver.recv()
                except (EOFError, OSError):
                    receiver.close()
                    break
                if progress is not None:
                    progress(message)

        process = context.Process(target=target, args=args, kwargs=kwargs or {})
        try:
            cancel_check()
            start(process)
            if channel:
                sender.close()
            while process.is_alive():
                cancel_check()
                drain()
                process.join(0.02)
            drain()
            cancel_check()
            return process.exitcode
        finally:
            if process.pid is not None:
                if process.is_alive() and cooperative:
                    cancelled.set()
                    process.join(0.25)
                if process.is_alive():
                    process.terminate()
                    process.join(0.5 if cooperative else 5)
                    if process.is_alive():
                        process.kill()
                process.join()
            process.close()
