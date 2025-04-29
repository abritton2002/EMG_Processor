@echo off
setlocal EnableDelayedExpansion

REM Set error codes
set ERROR_PYTHON_NOT_FOUND=1
set ERROR_REQUIREMENTS=4
set ERROR_DATA_DIR=5
set ERROR_PIPELINE=6
set ERROR_ENV_FILE=7

echo Starting EMG Pipeline...

REM Change to the project directory
cd /d "%~dp0"

REM Create logs directory if it doesn't exist
if not exist "logs" (
    mkdir "logs" 2>nul
    if errorlevel 1 (
        echo Error: Failed to create logs directory - permission denied
        pause
        exit /b 1
    )
)

REM Get current timestamp for log filename
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set TIMESTAMP=%datetime:~0,8%_%datetime:~8,6%
set LOG_FILE=logs\pipeline_run_%TIMESTAMP%.log

echo =============================================== > "%LOG_FILE%"
echo EMG Pipeline Run - %date% %time% >> "%LOG_FILE%"
echo =============================================== >> "%LOG_FILE%"

REM Check for .env file
if not exist ".env" (
    echo Error: .env file not found >> "%LOG_FILE%"
    echo Error: .env file not found. Please create a .env file with required database configuration.
    echo Required variables: DB_HOST, DB_USER, DB_PASSWORD, DB_NAME
    pause
    exit /b %ERROR_ENV_FILE%
)

REM Find Python - first try system path
echo Searching for Python installation... >> "%LOG_FILE%"
set PYTHON_CMD=python
set PYTHON_PATH_ADDED=0

REM Check if Python is in PATH already
where python > nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Found Python in system PATH >> "%LOG_FILE%"
    goto :PYTHON_FOUND
)

REM If not in PATH, search for it and add it temporarily
echo Python not in PATH, searching common locations... >> "%LOG_FILE%"

REM Check common installation locations
if exist "C:\Python312\python.exe" (
    set PYTHON_PATH=C:\Python312
    set PYTHON_CMD=!PYTHON_PATH!\python.exe
    goto :ADD_PYTHON_PATH
)

if exist "C:\Python311\python.exe" (
    set PYTHON_PATH=C:\Python311
    set PYTHON_CMD=!PYTHON_PATH!\python.exe
    goto :ADD_PYTHON_PATH
)

if exist "C:\Python310\python.exe" (
    set PYTHON_PATH=C:\Python310
    set PYTHON_CMD=!PYTHON_PATH!\python.exe
    goto :ADD_PYTHON_PATH
)

if exist "C:\Python39\python.exe" (
    set PYTHON_PATH=C:\Python39
    set PYTHON_CMD=!PYTHON_PATH!\python.exe
    goto :ADD_PYTHON_PATH
)

REM No Python found anywhere
echo Error: Python is not installed or not found in common locations >> "%LOG_FILE%"
echo Error: Python is not installed or not found in common locations
echo Please install Python 3.x from https://www.python.org/downloads/
pause
exit /b %ERROR_PYTHON_NOT_FOUND%

:ADD_PYTHON_PATH
REM Temporarily add Python to PATH for this session
echo Found Python at !PYTHON_PATH! >> "%LOG_FILE%"
echo Temporarily adding Python to PATH >> "%LOG_FILE%"
set PATH=!PYTHON_PATH!;!PYTHON_PATH!\Scripts;%PATH%
set PYTHON_PATH_ADDED=1
set PYTHON_CMD=python

:PYTHON_FOUND
echo Using Python: %PYTHON_CMD% >> "%LOG_FILE%"
%PYTHON_CMD% --version >> "%LOG_FILE%" 2>&1
%PYTHON_CMD% --version 2>&1

REM Check Python version is 3.x - more reliable method
%PYTHON_CMD% -c "import sys; print('PYTHON_VERSION=' + str(sys.version_info[0]))" > version.txt
set /p PYTHON_VER_LINE=<version.txt
del version.txt

echo Python version line: %PYTHON_VER_LINE% >> "%LOG_FILE%"
set PYTHON_VERSION=%PYTHON_VER_LINE:~15%

if not "%PYTHON_VERSION%"=="3" (
    echo Error: Python 3.x is required, found Python %PYTHON_VERSION%.x >> "%LOG_FILE%"
    echo Error: Python 3.x is required, found Python %PYTHON_VERSION%.x
    pause
    exit /b %ERROR_PYTHON_NOT_FOUND%
)

REM Install required packages directly (skip virtual environment)
echo Installing required packages with --user flag... >> "%LOG_FILE%"
echo This may take a few minutes...
%PYTHON_CMD% -m pip install --user -r requirements.txt >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    echo Warning: Failed to install packages with --user flag >> "%LOG_FILE%"
    echo Warning: Failed to install packages with --user flag
    echo Continuing anyway...
)

REM Set and validate the data directory
set DATA_DIR=M:\EMG_DATA_new
echo Validating data directory: %DATA_DIR% >> "%LOG_FILE%"

REM Check if data directory exists and is accessible
if not exist "%DATA_DIR%" (
    echo Warning: Data directory not found: %DATA_DIR% >> "%LOG_FILE%"
    echo Warning: Data directory not found: %DATA_DIR%
    echo The pipeline will try to use the current directory instead.
    set DATA_DIR=.
)

REM Run the pipeline with enhanced error handling
echo Running pipeline... >> "%LOG_FILE%"
echo Running pipeline...
echo This may take several minutes depending on the number of files...

%PYTHON_CMD% main.py >> "%LOG_FILE%" 2>&1
set PIPELINE_EXIT_CODE=%errorlevel%

REM Check pipeline exit code and provide specific error messages
if %PIPELINE_EXIT_CODE% neq 0 (
    echo Error: Pipeline failed with exit code %PIPELINE_EXIT_CODE% >> "%LOG_FILE%"
    echo Error: Pipeline failed with exit code %PIPELINE_EXIT_CODE%
    echo Check the log file for details: %LOG_FILE%
    pause
    exit /b %ERROR_PIPELINE%
) else (
    echo Pipeline completed successfully >> "%LOG_FILE%"
    echo Pipeline completed successfully
)

echo =============================================== >> "%LOG_FILE%"
echo Pipeline run completed at %date% %time% >> "%LOG_FILE%"
echo =============================================== >> "%LOG_FILE%"

REM Keep the window open to see any errors
echo.
echo Press any key to close this window...
pause > nul