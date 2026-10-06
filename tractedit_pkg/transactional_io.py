# -*- coding: utf-8 -*-
"""Transactional filesystem helpers for application exports."""

from __future__ import annotations

from contextlib import contextmanager
import errno
import os
from pathlib import Path
import sys
import tempfile
from typing import Callable, Iterator, Sequence


def _temporary_sibling(destination: Path, marker: str) -> Path:
    parent = destination.parent
    if not parent.is_dir():
        raise FileNotFoundError(f"Destination directory does not exist: {parent}")

    suffix = "".join(destination.suffixes)
    stem = destination.name[: -len(suffix)] if suffix else destination.name
    descriptor, temporary_name = tempfile.mkstemp(
        dir=parent,
        prefix=f".{stem}.{marker}-",
        suffix=suffix,
    )
    os.close(descriptor)
    return Path(temporary_name)


def _validate_staged_file(path: Path) -> None:
    if not path.is_file():
        raise OSError(f"Writer did not create the staged output: {path.name}")
    if path.stat().st_size == 0:
        raise OSError(f"Writer created an empty staged output: {path.name}")


def _remove_if_present(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _rename_exclusive(source: Path, destination: Path) -> None:
    """Native atomic rename without replacement, also on volumes without links."""
    if os.name == "nt":
        # Unlike POSIX rename, Windows rename refuses an existing destination.
        os.rename(source, destination)
        return

    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin" and hasattr(libc, "renamex_np"):
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        arguments = (os.fsencode(source), os.fsencode(destination), 4)  # RENAME_EXCL
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        # AT_FDCWD, RENAME_NOREPLACE; source and destination are siblings.
        arguments = (-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    else:
        raise OSError(errno.ENOTSUP, "Atomic exclusive rename is unavailable")
    rename.restype = ctypes.c_int
    if rename(*arguments) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(destination))


@contextmanager
def staged_new_output(destination: str | os.PathLike[str]) -> Iterator[Path]:
    """Publish a new file atomically, refusing to replace an existing path."""
    final_path = Path(destination)
    staged_path = _temporary_sibling(final_path, "tractedit-stage")
    try:
        yield staged_path
        _validate_staged_file(staged_path)
        if os.name == "nt":
            _rename_exclusive(staged_path, final_path)
        else:
            try:
                os.link(staged_path, final_path)
            except OSError as error:
                if error.errno not in {
                    errno.EPERM,
                    errno.ENOTSUP,
                    errno.EOPNOTSUPP,
                    errno.ENOSYS,
                }:
                    raise
                _rename_exclusive(staged_path, final_path)
    finally:
        _remove_if_present(staged_path)


@contextmanager
def staged_output(destination: str | os.PathLike[str]) -> Iterator[Path]:
    """Yield a sibling staging path and publish it on success."""
    final_path = Path(destination)
    staged_path = _temporary_sibling(final_path, "tractedit-stage")
    try:
        yield staged_path
        _validate_staged_file(staged_path)
        os.replace(staged_path, final_path)
    finally:
        _remove_if_present(staged_path)


@contextmanager
def staged_output_set(
    destinations: Sequence[str | os.PathLike[str]],
) -> Iterator[dict[Path, Path]]:
    """Stage a related file set and roll back if publication is incomplete."""
    final_paths = [Path(path) for path in destinations]
    if len(set(final_paths)) != len(final_paths):
        raise ValueError("Transactional output destinations must be unique.")

    staged: dict[Path, Path] = {}
    backups: dict[Path, Path] = {}
    committed: set[Path] = set()
    published = False

    try:
        for path in final_paths:
            staged[path] = _temporary_sibling(path, "tractedit-stage")
        yield staged
        for path in staged.values():
            _validate_staged_file(path)

        try:
            for final_path in final_paths:
                if final_path.exists():
                    backup = _temporary_sibling(final_path, "tractedit-backup")
                    try:
                        os.replace(final_path, backup)
                    except OSError:
                        _remove_if_present(backup)
                        raise
                    backups[final_path] = backup

            for final_path in final_paths:
                os.replace(staged[final_path], final_path)
                committed.add(final_path)
            published = True
        except BaseException as commit_error:
            rollback_errors = []
            for final_path in committed:
                try:
                    _remove_if_present(final_path)
                except OSError as error:
                    rollback_errors.append(error)
            for final_path, backup in backups.items():
                if backup.exists():
                    try:
                        os.replace(backup, final_path)
                    except OSError as error:
                        rollback_errors.append(error)
            if rollback_errors:
                retained = [str(path) for path in backups.values() if path.exists()]
                raise OSError(
                    "Output-set commit and rollback failed; inspect retained "
                    f"backups: {', '.join(retained) or 'none'}"
                ) from commit_error
            raise
    finally:
        for path in staged.values():
            _remove_if_present(path)
        if published:
            for backup in backups.values():
                _remove_if_present(backup)


def transactional_save(
    destination: str | os.PathLike[str],
    writer: Callable[[str], None],
) -> None:
    """Run ``writer`` against a staging path and publish its complete output."""
    with staged_output(destination) as staged_path:
        writer(str(staged_path))
