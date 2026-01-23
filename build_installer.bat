@echo off
setlocal enabledelayedexpansion

pushd "%~dp0"

echo [1/6] Checking Python...
where python >nul 2>nul
if errorlevel 1 (
  echo ERROR: Python not found. Install Python 3.10+ and retry.
  exit /b 1
)
python -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)"
if errorlevel 1 (
  echo ERROR: Python 3.10+ is required.
  exit /b 1
)

echo [2/6] Checking Inno Setup compiler...
where ISCC.exe >nul 2>nul
if errorlevel 1 (
  echo ERROR: ISCC.exe not found in PATH. Install Inno Setup and add ISCC.exe to PATH.
  exit /b 1
)

echo [3/6] Creating build virtual environment...
set VENV_DIR=.venv-build
if not exist "%VENV_DIR%" (
  python -m venv "%VENV_DIR%"
)
call "%VENV_DIR%\Scripts\activate.bat"

echo [4/6] Installing build dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pyinstaller

echo [5/6] Building single-file executable...
python packaging\generate_installer_vars.py
python -m PyInstaller --noconfirm packaging\ebsd_gui.spec
if not exist dist\EBSD_Scan_Comparator.exe (
  echo ERROR: PyInstaller did not produce dist\EBSD_Scan_Comparator.exe
  exit /b 1
)

echo [6/6] Building installer...
ISCC.exe packaging\installer.iss
if not exist dist\installer\EBSD_Scan_Comparator_Setup.exe (
  echo ERROR: Installer output not found at dist\installer\EBSD_Scan_Comparator_Setup.exe
  exit /b 1
)

echo Build complete: dist\installer\EBSD_Scan_Comparator_Setup.exe
popd
endlocal
