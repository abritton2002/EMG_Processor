@echo off
setlocal EnableDelayedExpansion

echo ========================================
echo   EMG Pipeline Execution
echo ========================================
echo.

REM Create logs directory if it doesn't exist
if not exist "logs" mkdir "logs" 2>nul

REM Get current timestamp for log filename
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set TIMESTAMP=%datetime:~0,8%_%datetime:~8,6%
set LOG_FILE=logs\pipeline_run_%TIMESTAMP%.log

echo =============================================== > "%LOG_FILE%"
echo EMG Pipeline Run - %date% %time% >> "%LOG_FILE%"
echo =============================================== >> "%LOG_FILE%"

REM Map M: drive (if not already mapped)
echo Mapping M: drive...
net use M: \\fl-fileshare.drivelinebaseball.com\edgeshare /user:readonly @@Driveline#1 >>"%LOG_FILE%" 2>>"%LOG_FILE%"

REM Set data directory to mapped location
set DATA_DIR=M:\EMG_DATA_new

echo Using data directory: %DATA_DIR% >> "%LOG_FILE%"
echo Using data directory: %DATA_DIR%

REM Now you can use %DATA_DIR%
if not exist "%DATA_DIR%" (
    echo ERROR: Data directory not found: %DATA_DIR% >> "%LOG_FILE%"
    echo ERROR: Data directory not found: %DATA_DIR%
    echo Check if network drive M: is correctly mapped.
    pause
    exit /b 1
)

REM Find Python in expected locations
set PYTHON_CMD=
if exist "C:\Python312\python.exe" (
    set PYTHON_CMD=C:\Python312\python.exe
) else if exist "C:\Python311\python.exe" (
    set PYTHON_CMD=C:\Python311\python.exe
) else if exist "C:\Python310\python.exe" (
    set PYTHON_CMD=C:\Python310\python.exe
) else (
    REM Try system path
    where python > nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        set PYTHON_CMD=python
    ) else (
        echo ERROR: Python not found >> "%LOG_FILE%"
        echo ERROR: Python not found. Please install Python 3.x.
        pause
        exit /b 1
    )
)

echo Using Python: %PYTHON_CMD% >> "%LOG_FILE%"
echo Using Python: %PYTHON_CMD%

REM Verify Python version
%PYTHON_CMD% --version >> "%LOG_FILE%" 2>&1
%PYTHON_CMD% --version

REM Ensure dependencies are installed
echo Installing required packages... >> "%LOG_FILE%"
%PYTHON_CMD% -m pip install -r requirements.txt >> "%LOG_FILE%" 2>&1

echo ---------------------------------------- >> "%LOG_FILE%"
echo Testing and initializing database tables... >> "%LOG_FILE%"
echo Testing and initializing database tables...

REM Run DB setup script to ensure tables are created
%PYTHON_CMD% -c "from db_connector import DBConnector; db = DBConnector(); db.test_connection(); db.create_tables()" >> "%LOG_FILE%" 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to create database tables >> "%LOG_FILE%"
    echo ERROR: Failed to create database tables.
    echo Check your database connection settings in .env file.
    pause
    exit /b 1
)

echo Database tables initialized successfully >> "%LOG_FILE%"
echo Database tables initialized successfully
echo ---------------------------------------- >> "%LOG_FILE%"

echo Running EMG pipeline... >> "%LOG_FILE%"
echo Running EMG pipeline...

REM Run the pipeline using the data directory
%PYTHON_CMD% main.py -d "%DATA_DIR%" -r >> "%LOG_FILE%" 2>&1
set PIPELINE_EXIT_CODE=%errorlevel%

if %PIPELINE_EXIT_CODE% neq 0 (
    echo Error: Pipeline failed with exit code %PIPELINE_EXIT_CODE% >> "%LOG_FILE%"
    echo Error: Pipeline failed with exit code %PIPELINE_EXIT_CODE%
    echo Check the log file for details: %LOG_FILE%
) else (
    echo Pipeline completed successfully >> "%LOG_FILE%"
    echo Pipeline completed successfully
)

echo =============================================== >> "%LOG_FILE%"
echo Pipeline run completed at %date% %time% >> "%LOG_FILE%"
echo =============================================== >> "%LOG_FILE%"

echo.
echo Done! Check the log file for details: %LOG_FILE%
pause