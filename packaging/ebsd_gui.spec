# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the EBSD Scan Comparator GUI."""

from __future__ import annotations

from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from app_metadata import (
    APP_DESCRIPTION,
    APP_EXE_NAME,
    APP_ICON_PATH,
    APP_NAME,
    APP_COPYRIGHT,
    APP_PUBLISHER,
    APP_VERSION,
)


block_cipher = None


def _parse_version(version: str) -> tuple[int, int, int, int]:
    """Parse a dotted version string into a 4-part tuple.

    Parameters:
        version: Version string like "1.2.3".

    Returns:
        Tuple of integers suitable for Windows version resources.
    """

    parts = (version.split(".") + ["0", "0", "0", "0"])[:4]
    parsed = []
    for part in parts:
        parsed.append(int(part) if part.isdigit() else 0)
    return tuple(parsed)  # type: ignore[return-value]


def _write_version_file(path: Path) -> None:
    """Write a Windows version resource file.

    Parameters:
        path: Destination path for the version resource file.

    Returns:
        None.
    """

    file_version = _parse_version(APP_VERSION)
    product_version = file_version
    version_info = f"""
# UTF-8
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers={file_version},
    prodvers={product_version},
    mask=0x3F,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo(
      [
        StringTable(
          '040904B0',
          [
            StringStruct('CompanyName', '{APP_PUBLISHER}'),
            StringStruct('FileDescription', '{APP_DESCRIPTION}'),
            StringStruct('FileVersion', '{APP_VERSION}'),
            StringStruct('InternalName', '{APP_NAME}'),
            StringStruct('LegalCopyright', '{APP_COPYRIGHT}'),
            StringStruct('OriginalFilename', '{APP_EXE_NAME}'),
            StringStruct('ProductName', '{APP_NAME}'),
            StringStruct('ProductVersion', '{APP_VERSION}')
          ]
        )
      ]
    ),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
"""
    path.write_text(version_info.strip() + "\n", encoding="utf-8")


version_file = repo_root / "packaging" / "version_info.txt"
_write_version_file(version_file)

icon_path = repo_root / APP_ICON_PATH

hidden_imports = [
    "h5py",
    "numpy",
    "matplotlib",
    "matplotlib.backends.backend_qtagg",
    "PySide6",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "scipy",
    "skimage",
    "sklearn",
    "numba",
    "pandas",
    "yaml",
    "kikuchipy",
    "hyperspy",
    "orix",
    "pyxem",
    "cv2",
    "natsort",
    "loguru",
]


data_files = [
    (str(repo_root / "testData"), "testData"),
    (str(repo_root / "configs"), "configs"),
    (str(repo_root / "assets" / "icons"), "assets/icons"),
]


app_name = Path(APP_EXE_NAME).stem


analysis = Analysis(
    [str(repo_root / "kikuchiBandAnalyzer" / "ebsd_compare" / "gui" / "main_window.py")],
    pathex=[str(repo_root)],
    binaries=[],
    datas=data_files,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(analysis.pure, analysis.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    analysis.scripts,
    analysis.binaries,
    analysis.datas,
    [],
    name=app_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=str(icon_path),
    version=str(version_file),
)
