@echo off
setlocal
cd /d "%~dp0"
title DocuMind

rem Use a virtual environment if there is one: DOCUMIND_VENV, then .venv, venv and the
rem per-user folder recommended when the project is in a cloud-synced folder.
set "VENV="
if defined DOCUMIND_VENV if exist "%DOCUMIND_VENV%\Scripts\activate.bat" set "VENV=%DOCUMIND_VENV%"
if not defined VENV if exist ".venv\Scripts\activate.bat" set "VENV=.venv"
if not defined VENV if exist "venv\Scripts\activate.bat" set "VENV=venv"
if not defined VENV if exist "%LOCALAPPDATA%\DocuMind\.venv\Scripts\activate.bat" set "VENV=%LOCALAPPDATA%\DocuMind\.venv"

rem No brackets around this, because a ")" in a folder name would end the block early.
if not defined VENV goto no_venv
echo Using the virtual environment in "%VENV%"
call "%VENV%\Scripts\activate.bat"
goto check_python

:no_venv
echo No virtual environment found, so using the Python on this computer.

:check_python
rem Run Python rather than only looking for it, because Windows has a "python" shortcut
rem that opens the Microsoft Store when Python is not installed.
python --version >nul 2>nul
if errorlevel 1 (
    echo.
    echo ERROR: Python was not found.
    echo Install Python 3.12 from https://www.python.org/downloads/ and tick
    echo "Add python.exe to PATH", then run this file again.
    goto failed
)

python -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)"
if errorlevel 1 (
    echo.
    echo ERROR: DocuMind needs Python 3.12 or newer. This computer is using:
    python --version
    goto failed
)

python -c "import importlib.util as u, sys; names = 'streamlit plotly pymupdf docx sentence_transformers faiss sklearn pandas requests reportlab'.split(); missing = [n for n in names if u.find_spec(n) is None]; print('Missing packages: ' + ', '.join(missing)) if missing else None; sys.exit(1 if missing else 0)"
if errorlevel 1 (
    echo.
    echo ERROR: Some required packages are not installed. Install them with:
    echo     python -m pip install -r requirements.txt
    goto failed
)

echo Starting DocuMind. It opens in your browser. Close this window to stop it.
python -m streamlit run app.py %*
if errorlevel 1 (
    echo.
    echo ERROR: DocuMind stopped unexpectedly. See the messages above.
    goto failed
)
exit /b 0

:failed
if not defined DOCUMIND_NO_PAUSE pause
exit /b 1
