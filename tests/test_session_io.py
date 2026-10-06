"""Small archive contracts independent of Qt and VTK."""

import json
import errno
import re
import zipfile

import numpy as np
import pytest

from tractedit_pkg.session_io import (
    SessionCancelled,
    SessionError,
    fingerprint,
    read_session,
    timestamped_path,
    write_session,
)


def test_numeric_round_trip_preserves_values_order_and_types(tmp_path):
    path = timestamped_path(str(tmp_path / "work"))
    values = np.array([0.25, -0.0, np.nan, np.inf, 256], dtype=np.float64)
    document = {
        "sources": {},
        "state": {"roi": values, "ids": {7, 2}, "history": [{"tuple": (2, 3)}]},
        "views": {4: True},
    }
    write_session(path, document)
    restored = read_session(path)
    assert restored["state"]["roi"].tobytes() == values.tobytes()
    assert restored["state"]["ids"] == {2, 7}
    assert restored["state"]["history"] == [{"tuple": (2, 3)}]
    assert restored["views"] == {4: True}


def test_numpy_integer_set_uses_compact_array_encoding(tmp_path):
    path = tmp_path / "numpy_indices.tractedit-session"
    values = {np.int64(1), np.uint32(3)}
    write_session(path, {"sources": {}, "state": values, "views": {}})
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    assert '"intset"' in json.dumps(manifest["payload"])
    assert read_session(path)["state"] == {1, 3}


def test_integer_set_outside_int64_range_keeps_its_value(tmp_path):
    path = tmp_path / "large_integer.tractedit-session"
    value = np.uint64(2**63)
    write_session(path, {"sources": {}, "state": {value}, "views": {}})
    assert read_session(path)["state"] == {2**63}


def test_timestamp_precedes_extension_and_does_not_accumulate(tmp_path):
    first = timestamped_path(str(tmp_path / "work.tractedit-session"))
    first.touch()
    second = timestamped_path(str(first))
    expected = r"work_\d{8}_\d{6}_\d{6}(?:_\d+)?\.tractedit-session"
    assert re.fullmatch(expected, first.name)
    assert re.fullmatch(expected, second.name)
    assert first != second

    legacy = timestamped_path(
        str(tmp_path / "work.tractedit-session-20260924_143025_123456")
    )
    assert re.fullmatch(expected, legacy.name)


def test_publication_never_overwrites_existing_file(tmp_path):
    path = tmp_path / "existing"
    path.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        write_session(path, {"sources": {}, "state": {}, "views": {}})
    assert path.read_bytes() == b"keep"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize(
    "prefix", ["/home/user/", "/Users/user/", "C:\\data\\", "\\\\server\\share\\"]
)
@pytest.mark.parametrize("schema", [1, 2])
@pytest.mark.parametrize("relative", [False, True])
def test_relocate_source_from_another_os(tmp_path, prefix, schema, relative):
    directory = tmp_path / "data" if relative else tmp_path
    directory.mkdir(exist_ok=True)
    source = directory / "fascio è 01.trk"
    source.write_bytes(b"source geometry")
    record = fingerprint(source)
    record["path"] = prefix + source.name
    path = tmp_path / "portable.tractedit-session"
    original = tmp_path / "original.tractedit-session"
    write_session(
        original, {"sources": {"tractogram": record}, "state": {}, "views": {}}
    )
    with zipfile.ZipFile(original) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    manifest["schema"] = schema
    manifest["relative_sources"] = (
        {"tractogram": "data/" + source.name} if relative else {}
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
    assert read_session(path)["sources"]["tractogram"] == fingerprint(source)
    source.write_bytes(b"changed content")
    with pytest.raises(SessionError, match="changed"):
        read_session(path)


@pytest.mark.parametrize("existing", [False, True])
def test_publication_without_hardlinks(tmp_path, monkeypatch, existing):
    from tractedit_pkg import transactional_io

    def unsupported(*args):
        raise OSError(errno.EOPNOTSUPP, "Hard links unsupported")

    monkeypatch.setattr(transactional_io.os, "link", unsupported)
    path = tmp_path / "session è.tractedit-session"
    if existing:
        path.write_bytes(b"keep")
        with pytest.raises(FileExistsError):
            write_session(path, {"sources": {}, "state": {}, "views": {}})
        assert path.read_bytes() == b"keep"
    else:
        write_session(path, {"sources": {}, "state": {}, "views": {}})
        assert read_session(path)["state"] == {}
    assert list(tmp_path.iterdir()) == [path]


def test_cancellation_removes_partial_archive(tmp_path):
    path = tmp_path / "cancelled"
    with pytest.raises(SessionCancelled):
        write_session(
            path, {"sources": {}, "state": {}, "views": {}}, cancelled=lambda: True
        )
    assert list(tmp_path.iterdir()) == []


def test_source_changed_without_size_change_is_rejected(tmp_path):
    source = tmp_path / "input"
    source.write_bytes(b"original")
    path = tmp_path / "session"
    write_session(
        path,
        {"sources": {"tractogram": fingerprint(str(source))}, "state": {}, "views": {}},
    )
    source.write_bytes(b"modified")
    with pytest.raises(SessionError, match="changed"):
        read_session(path)


def test_object_arrays_never_serialize(tmp_path):
    path = tmp_path / "session"
    with pytest.raises(SessionError, match="numeric"):
        write_session(
            path, {"sources": {}, "state": {"bad": np.array([object()])}, "views": {}}
        )
    assert not path.exists()


@pytest.mark.parametrize("name", ["../outside", "/absolute", "arrays/../0.npy"])
def test_unsafe_members_are_rejected(tmp_path, name):
    path = tmp_path / "bad"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps({}))
        archive.writestr(name, b"bad")
    with pytest.raises(SessionError, match="Invalid archive member"):
        read_session(path)


def test_duplicate_archive_members_are_rejected(tmp_path):
    path = tmp_path / "duplicate"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", "{}")
        with pytest.warns(UserWarning):
            archive.writestr("manifest.json", "{}")
    with pytest.raises(SessionError, match="Duplicate"):
        read_session(path)


def test_forged_array_shape_rejected_before_allocation(tmp_path, monkeypatch):
    import io

    path = tmp_path / "forged"
    valid = tmp_path / "valid"
    write_session(valid, {"sources": {}, "state": np.zeros(1), "views": {}})
    with zipfile.ZipFile(valid) as archive:
        manifest = archive.read("manifest.json")
    buffer = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        buffer, {"descr": "<f8", "fortran_order": False, "shape": (10**12,)}
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", manifest)
        archive.writestr("arrays/0.npy", buffer.getvalue())
    monkeypatch.setattr(
        np, "empty", lambda *a, **kw: pytest.fail("Allocated forged shape")
    )
    with pytest.raises(SessionError, match="shape"):
        read_session(path)


def test_schema_version_rejected(tmp_path):
    path = tmp_path / "future"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "manifest.json", json.dumps({"format": "tractedit-session", "schema": 999})
        )
    with pytest.raises(SessionError, match="schema"):
        read_session(path)


def test_app_version_is_provenance_not_a_restore_requirement(tmp_path):
    original = tmp_path / "original"
    changed = tmp_path / "changed"
    write_session(original, {"sources": {}, "state": {"value": 7}, "views": {}})
    with zipfile.ZipFile(original) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    manifest["application_version"] = "99.0.0"
    with zipfile.ZipFile(changed, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))

    assert read_session(changed)["state"] == {"value": 7}


def test_source_destination_alias_rejected(tmp_path):
    path = tmp_path / "source"
    path.write_bytes(b"data")
    with pytest.raises(SessionError, match="overwrite"):
        write_session(
            path,
            {
                "sources": {"tractogram": fingerprint(str(path))},
                "state": {},
                "views": {},
            },
        )
    assert path.read_bytes() == b"data"
