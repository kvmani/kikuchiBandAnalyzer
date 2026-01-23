# Windows Installer Guide (EBSD Scan Comparator)

This guide explains how to build a professional Windows installer that bundles the EBSD Scan Comparator GUI into a single setup EXE. End users do not need Python.

## What this installer does

- Builds a single-file GUI executable with PyInstaller.
- Wraps it in a modern Windows installer (Inno Setup).
- Creates Start Menu and optional Desktop shortcuts.
- Installs configs and test data alongside the app.

## Requirements (one-time setup)

1. Install Python 3.10 or newer (64-bit)
   - Download: https://www.python.org/downloads/windows/
   - During install, check "Add Python to PATH".

2. Install Inno Setup (64-bit)
   - Download: https://jrsoftware.org/isdl.php
   - During install, check "Add Inno Setup to PATH" or add `ISCC.exe` to PATH manually.

3. Optional: Git (if you are cloning the repo)
   - Download: https://git-scm.com/download/win

## One-click build (recommended)

From the repository root, run:

```
build_installer.bat
```

The script will:
1. Create a build virtual environment (`.venv-build`).
2. Install dependencies.
3. Build the single-file EXE with PyInstaller.
4. Create the installer with Inno Setup.

## Output location

After a successful build, the installer is located at:

```
dist\installer\EBSD_Scan_Comparator_Setup.exe
```

This is the file you can share with Windows users.

## How to test the installer

1. Copy `dist\installer\EBSD_Scan_Comparator_Setup.exe` to a clean Windows machine.
2. Run the installer and choose the default options.
3. Confirm:
   - The app launches from Start Menu.
   - Desktop shortcut works (if selected).
   - The app opens without Python installed.
   - You can browse to `{install_dir}\testData` and load a sample scan.

## How to update the version in the future

1. Open `app_metadata.py`.
2. Update `APP_VERSION`.
3. Re-run:

```
build_installer.bat
```

The version is pulled automatically into both the EXE and the installer.

## Common errors and fixes

1. "Python not found"
   - Install Python 3.10+ and check "Add Python to PATH".

2. "ISCC.exe not found in PATH"
   - Reinstall Inno Setup and enable PATH integration, or add the folder containing `ISCC.exe` to PATH.

3. "PyInstaller did not produce dist\EBSD_Scan_Comparator.exe"
   - Ensure dependencies installed correctly. Re-run the script and check for errors above this line.

4. Antivirus blocks the EXE
   - Sign the EXE with a trusted code-signing certificate for distribution.

## Notes

- The installer supports per-user installs by default, with an optional admin install option.
- The setup EXE is a single file; the installed app is a normal Windows application with shortcuts.
- If you need an offline build, run the script once on a machine with internet to populate the pip cache, or point pip to a local wheelhouse.
