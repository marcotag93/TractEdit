"""Regression tests for the TractEdit command-line interface."""

import sys

import pytest

import main as tractedit_main
from tractedit_pkg import __version__


@pytest.mark.parametrize("flag", ["-v", "-V", "--version"])
def test_version_flags_exit_cleanly_and_print_current_version(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    flag: str,
) -> None:
    monkeypatch.setattr(sys, "argv", ["tractedit", flag])

    with pytest.raises(SystemExit) as exc_info:
        tractedit_main.main()

    assert exc_info.value.code == 0
    assert f"Version: {__version__}" in capsys.readouterr().out
