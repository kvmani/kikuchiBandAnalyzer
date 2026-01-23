"""Generate Inno Setup variables from application metadata."""

from __future__ import annotations

from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import app_metadata as metadata


def _build_lines() -> list[str]:
    """Build preprocessor definition lines for Inno Setup.

    Returns:
        List of preprocessor definition strings.
    """

    return [
        f'#define AppName "{metadata.APP_NAME}"',
        f'#define AppVersion "{metadata.APP_VERSION}"',
        f'#define AppPublisher "{metadata.APP_PUBLISHER}"',
        f'#define AppURL "{metadata.APP_WEBSITE}"',
        f'#define AppExeName "{metadata.APP_EXE_NAME}"',
        f'#define AppId "{metadata.APP_ID}"',
    ]


def main() -> None:
    """Write the installer variables file used by Inno Setup.

    Returns:
        None.
    """

    output_path = Path(__file__).resolve().parent / "installer_vars.iss"
    output_path.write_text("\n".join(_build_lines()) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
