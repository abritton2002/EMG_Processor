@echo off
echo EMG Data Path Diagnostic Tool
echo =============================
echo.

REM Check if M: drive exists at all
if exist "M:\" (
    echo M: drive is accessible
) else (
    echo M: drive is NOT accessible
)
echo.

REM Check specific path
if exist "M:\EMG_DATA_new" (
    echo M:\EMG_DATA_new folder exists
) else (
    echo M:\EMG_DATA_new folder does NOT exist
)
echo.

REM List contents of M: drive
echo Contents of M: drive root:
dir M:\ /B

echo.
echo =============================
echo Checking alternative locations:
echo.

REM Check for EMG_DATA_new on other drives
for %%D in (N: O: P: Q: R: S: T: U: V: W: X: Y: Z:) do (
    if exist "%%D\" (
        echo %%D drive is accessible
        if exist "%%D\EMG_DATA_new" (
            echo FOUND: %%D\EMG_DATA_new exists!
        )
    )
)

echo.
echo Diagnostic complete. Press any key to exit.
pause > nul