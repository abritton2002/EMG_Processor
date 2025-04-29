@echo off
setlocal EnableDelayedExpansion

REM Set error codes
set ERROR_PYTHON_NOT_FOUND=1
set ERROR_VENV_CREATION=2
set ERROR_VENV_ACTIVATION=3
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

REM Check Python version
echo Checking Python installation... >> "%LOG_FILE%"
python --version > nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH >> "%LOG_FILE%"
    echo Error: Python is not installed or not in PATH
    echo Please install Python and add it to your system PATH
    pause
    exit /b %ERROR_PYTHON_NOT_FOUND%
)

REM Check Python version is 3.x
for /f "tokens=2" %%I in ('python -c "import sys; print(sys.version_info[0])"') do set PYTHON_VERSION=%%I
if not "%PYTHON_VERSION%"=="3" (
    echo Error: Python 3.x is required, found Python %PYTHON_VERSION%.x >> "%LOG_FILE%"
    echo Error: Python 3.x is required, found Python %PYTHON_VERSION%.x
    pause
    exit /b %ERROR_PYTHON_NOT_FOUND%
)

REM Check if virtual environment exists, if not create it
if not exist ".venv" (
    echo Creating virtual environment... >> "%LOG_FILE%"
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo Error: Failed to create virtual environment >> "%LOG_FILE%"
        echo Error: Failed to create virtual environment
        echo Please check Python installation and permissions
        pause
        exit /b %ERROR_VENV_CREATION%
    )
)

REM Activate virtual environment with error handling
echo Activating virtual environment... >> "%LOG_FILE%"
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate
) else (
    echo Error: Virtual environment activation script not found >> "%LOG_FILE%"
    echo Error: Virtual environment activation script not found
    echo Try deleting the .venv directory and running the script again
    pause
    exit /b %ERROR_VENV_ACTIVATION%
)

REM Verify virtual environment activation
python -c "import sys; print(sys.prefix)" | findstr /i ".venv" > nul
if errorlevel 1 (
    echo Error: Failed to activate virtual environment >> "%LOG_FILE%"
    echo Error: Failed to activate virtual environment
    pause
    exit /b %ERROR_VENV_ACTIVATION%
)

REM Install/update requirements with error handling
echo Checking/updating requirements... >> "%LOG_FILE%"
echo This may take a few minutes...
pip install -r requirements.txt >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    echo Error: Failed to install requirements >> "%LOG_FILE%"
    echo Error: Failed to install requirements
    echo Check the log file for details: %LOG_FILE%
    pause
    exit /b %ERROR_REQUIREMENTS%
)

REM Set and validate the data directory
set DATA_DIR=M:\EMG_DATA_new
echo Validating data directory: %DATA_DIR% >> "%LOG_FILE%"

REM Check if data directory exists and is accessible
if not exist "%DATA_DIR%" (
    echo Error: Data directory not found: %DATA_DIR% >> "%LOG_FILE%"
    echo Error: Data directory not found: %DATA_DIR%
    pause
    exit /b %ERROR_DATA_DIR%
)

REM Try to create a test file to verify write permissions
echo Testing write permissions... >> "%LOG_FILE%"
echo test > "%DATA_DIR%\test_permissions.tmp" 2>nul
if errorlevel 1 (
    echo Warning: No write permissions in data directory >> "%LOG_FILE%"
    echo Warning: No write permissions in data directory
    echo The pipeline will continue but may fail if writing is required
)
if exist "%DATA_DIR%\test_permissions.tmp" del "%DATA_DIR%\test_permissions.tmp"

REM Run the pipeline with enhanced error handling
echo Running pipeline for directory: %DATA_DIR% >> "%LOG_FILE%"
echo Running pipeline for directory: %DATA_DIR%
echo This may take several minutes depending on the number of files...

python main.py >> "%LOG_FILE%" 2>&1
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