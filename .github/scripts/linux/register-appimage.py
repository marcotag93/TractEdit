"""Install this AppImage's Open With entry for the current desktop user."""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tractedit_pkg.linux_appimage_integration import (  # noqa: E402
    register,
    run_cli,
)

__all__ = ["register", "run_cli"]

if __name__ == "__main__":
    raise SystemExit(run_cli())
