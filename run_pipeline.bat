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

REM Set Python to unbuffered mode for real-time output
set PYTHONUNBUFFERED=1

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
        echo ERROR: Python not found
        echo ERROR: Python not found >> "%LOG_FILE%"
        echo ERROR: Python not found. Please install Python 3.x.
        pause
        exit /b 1
    )
)

echo Using Python: %PYTHON_CMD%
echo Using Python: %PYTHON_CMD% >> "%LOG_FILE%"

REM Create a simple tee script to log both to console and file
echo import sys, os > tee.py
echo logfile = sys.argv[1] >> tee.py
echo for line in sys.stdin: >> tee.py
echo     sys.stdout.write(line) >> tee.py
echo     sys.stdout.flush() >> tee.py
echo     with open(logfile, 'a', encoding='utf-8') as f: >> tee.py
echo         f.write(line) >> tee.py
echo         f.flush() >> tee.py

REM Map M: drive (if not already mapped)
echo Mapping M: drive...
echo Mapping M: drive... >> "%LOG_FILE%"
net use M: \\fl-fileshare.drivelinebaseball.com\edgeshare /user:readonly @@Driveline#1 > temp.log 2>&1
type temp.log
type temp.log >> "%LOG_FILE%"

REM Set data directory to mapped location
set DATA_DIR=M:\EMG_DATA_new

echo Using data directory: %DATA_DIR%
echo Using data directory: %DATA_DIR% >> "%LOG_FILE%"

REM Now you can use %DATA_DIR%
if not exist "%DATA_DIR%" (
    echo ERROR: Data directory not found: %DATA_DIR%
    echo ERROR: Data directory not found: %DATA_DIR% >> "%LOG_FILE%"
    echo Check if network drive M: is correctly mapped.
    pause
    exit /b 1
)

REM Create a .env file for Slack if it doesn't exist
if not exist ".env" (
    echo Creating .env file...
    echo SLACK_BOT_TOKEN=xoxb-your-bot-token-here > .env
    echo SLACK_CHANNEL_ID=your-channel-id-here >> .env
    echo.
    echo IMPORTANT: Edit the .env file with your actual Slack token and channel ID
    echo.
)

REM Verify Python version
%PYTHON_CMD% --version > temp.log 2>&1
type temp.log
type temp.log >> "%LOG_FILE%"

REM Ensure dependencies are installed
echo Installing required packages...
echo Installing required packages... >> "%LOG_FILE%"
%PYTHON_CMD% -m pip install -r requirements.txt > temp.log 2>&1
type temp.log
type temp.log >> "%LOG_FILE%"

echo ----------------------------------------
echo ---------------------------------------- >> "%LOG_FILE%"
echo Testing and initializing database tables...
echo Testing and initializing database tables... >> "%LOG_FILE%"

REM Run the program with the --test-db flag to check database connection
%PYTHON_CMD% main.py --test-db > temp.log 2>&1
type temp.log
type temp.log >> "%LOG_FILE%"
set DB_TEST_CODE=%ERRORLEVEL%

if %DB_TEST_CODE% NEQ 0 (
    echo ERROR: Failed to connect to database
    echo ERROR: Failed to connect to database >> "%LOG_FILE%"
    echo Check your database connection settings in .env file.
    pause
    exit /b 1
)

echo Database connection successful
echo Database connection successful >> "%LOG_FILE%"
echo ----------------------------------------
echo ---------------------------------------- >> "%LOG_FILE%"

REM Create a batch file to run the main script and capture output
echo @echo off > run_and_log.bat
echo %PYTHON_CMD% main.py -d "%DATA_DIR%" -r ^> temp_pipeline.log 2^>^&1 >> run_and_log.bat
echo set PIPELINE_EXIT_CODE=%%ERRORLEVEL%% >> run_and_log.bat
echo type temp_pipeline.log >> run_and_log.bat
echo type temp_pipeline.log ^>^> "%LOG_FILE%" >> run_and_log.bat
echo exit /b %%PIPELINE_EXIT_CODE%% >> run_and_log.bat

echo Running EMG pipeline...
echo Running EMG pipeline... >> "%LOG_FILE%"

REM Execute the batch file we just created
call run_and_log.bat
set PIPELINE_EXIT_CODE=%ERRORLEVEL%

if %PIPELINE_EXIT_CODE% NEQ 0 (
    echo Error: Pipeline failed with exit code %PIPELINE_EXIT_CODE%
    echo Error: Pipeline failed with exit code %PIPELINE_EXIT_CODE% >> "%LOG_FILE%"
    echo Check the log file for details: %LOG_FILE%
) else (
    echo Pipeline completed successfully
    echo Pipeline completed successfully >> "%LOG_FILE%"
)

REM Send summary to Slack
echo Sending summary to Slack...
echo Sending summary to Slack... >> "%LOG_FILE%"
%PYTHON_CMD% summarize_and_send_log.py "%LOG_FILE%" > temp_slack.log 2>&1
type temp_slack.log
type temp_slack.log >> "%LOG_FILE%"
set SLACK_EXIT_CODE=%ERRORLEVEL%

if %SLACK_EXIT_CODE% NEQ 0 (
    echo Warning: Failed to send summary to Slack.
    echo Warning: Failed to send summary to Slack. >> "%LOG_FILE%"
    echo Check the error messages above for details.
) else (
    echo Summary successfully sent to Slack.
    echo Summary successfully sent to Slack. >> "%LOG_FILE%"
)

REM Clean up temporary files
del tee.py 2>nul
del temp.log 2>nul
del temp_pipeline.log 2>nul
del temp_slack.log 2>nul
del run_and_log.bat 2>nul

echo ===============================================
echo =============================================== >> "%LOG_FILE%"
echo Pipeline run completed at %date% %time%
echo Pipeline run completed at %date% %time% >> "%LOG_FILE%"
echo ===============================================
echo =============================================== >> "%LOG_FILE%"

echo.
echo Done! Check the log file for details: %LOG_FILE%
set SUMMARY_FILE=%LOG_FILE:~0,-4%_summary.txt
if exist "%SUMMARY_FILE%" (
    echo.
    echo Summary:
    echo.
    type "%SUMMARY_FILE%"
)
pause