@echo off
REM Change to the project directory
cd /d "%~dp0"

REM OPTIONAL: Activate virtual environment if you use one
REM call .venv\Scripts\activate

REM Set the data directory you want to process
set DATA_DIR=C:\Path\To\Your\Data

REM Run the pipeline (edit arguments as needed)
python main.py --directory "%DATA_DIR%" --recursive > pipeline_output.log 2>&1

REM Optional: Pause so you can see any error messages if running manually
REM pause 