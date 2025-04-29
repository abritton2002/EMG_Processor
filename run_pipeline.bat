@echo off
echo Starting EMG Pipeline...

REM Change to the project directory
cd /d "%~dp0"

REM Check if Python is installed
python --version > nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if virtual environment exists, if not create it
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Activate virtual environment
call .venv\Scripts\activate
if errorlevel 1 (
    echo Error: Failed to activate virtual environment
    pause
    exit /b 1
)

REM Install/update requirements if needed
pip install -r requirements.txt
if errorlevel 1 (
    echo Error: Failed to install requirements
    pause
    exit /b 1
)

REM Set the data directory
set DATA_DIR=M:\EMG_DATA_new

REM Check if data directory exists
if not exist "%DATA_DIR%" (
    echo Error: Data directory not found: %DATA_DIR%
    pause
    exit /b 1
)

REM Run the pipeline and log output
echo Running pipeline for directory: %DATA_DIR%
python main.py > pipeline_output.log 2>&1

REM Check if pipeline ran successfully
if errorlevel 1 (
    echo Error: Pipeline failed. Check pipeline_output.log for details
) else (
    echo Pipeline completed successfully
)

REM Keep the window open to see any errors
pause 