@echo off
SETLOCAL EnableDelayedExpansion

REM Check for admin rights
NET SESSION >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo This script requires administrator privileges.
    echo Please right-click and select "Run as administrator".
    pause
    exit /b
)

echo CUDA PATH Setup Tool
echo -------------------
echo.

REM Find CUDA installations
set "CUDA_BASE=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA"
set FOUND=0
set INDEX=0

if not exist "%CUDA_BASE%" (
    echo No CUDA installations found.
    echo Please install CUDA from https://developer.nvidia.com/cuda-downloads
    pause
    exit /b
)

REM List all CUDA versions
echo Found CUDA installations:
for /d %%G in ("%CUDA_BASE%\v*") do (
    set /a INDEX+=1
    set "CUDA_PATH[!INDEX!]=%%G"
    for /f "tokens=2 delims=v" %%H in ("%%~nxG") do (
        set "CUDA_VERSION[!INDEX!]=%%H"
    )
    echo !INDEX!. CUDA !CUDA_VERSION[%INDEX%]! at "!CUDA_PATH[%INDEX%]!"
    set FOUND=1
)

if %FOUND% EQU 0 (
    echo No CUDA installations found.
    echo Please install CUDA from https://developer.nvidia.com/cuda-downloads
    pause
    exit /b
)

REM Select CUDA version if multiple found
set SELECTED=1
if %INDEX% GTR 1 (
    echo.
    set /p SELECTED="Enter the number of the CUDA version to add to PATH: "
)

if !SELECTED! LSS 1 (
    set SELECTED=1
)
if !SELECTED! GTR %INDEX% (
    set SELECTED=%INDEX%
)

echo.
echo Selected CUDA !CUDA_VERSION[%SELECTED%]! at "!CUDA_PATH[%SELECTED%]!"
echo.

REM Paths to add
set "BIN_PATH=!CUDA_PATH[%SELECTED%]!\bin"
set "LIBNVVP_PATH=!CUDA_PATH[%SELECTED%]!\libnvvp"
set "CUPTI_PATH=!CUDA_PATH[%SELECTED%]!\extras\CUPTI\lib64"

REM Check if paths exist
set VALID_PATHS=0
if exist "!BIN_PATH!" (
    set /a VALID_PATHS+=1
)
if exist "!LIBNVVP_PATH!" (
    set /a VALID_PATHS+=1
)
if exist "!CUPTI_PATH!" (
    set /a VALID_PATHS+=1
)

if %VALID_PATHS% EQU 0 (
    echo No valid paths found in the selected CUDA installation.
    pause
    exit /b
)

REM Add paths to system PATH
echo Adding CUDA paths to system PATH...

REM Use SETX to set system environment variables
if exist "!BIN_PATH!" (
    echo Adding: !BIN_PATH!
    REM Check if path already exists in PATH
    call :CheckPath "!BIN_PATH!"
    if !ERRORLEVEL! NEQ 0 (
        setx PATH "%PATH%;!BIN_PATH!" /M
    ) else (
        echo Path already exists in PATH.
    )
)

if exist "!LIBNVVP_PATH!" (
    echo Adding: !LIBNVVP_PATH!
    call :CheckPath "!LIBNVVP_PATH!"
    if !ERRORLEVEL! NEQ 0 (
        setx PATH "%PATH%;!LIBNVVP_PATH!" /M
    ) else (
        echo Path already exists in PATH.
    )
)

if exist "!CUPTI_PATH!" (
    echo Adding: !CUPTI_PATH!
    call :CheckPath "!CUPTI_PATH!"
    if !ERRORLEVEL! NEQ 0 (
        setx PATH "%PATH%;!CUPTI_PATH!" /M
    ) else (
        echo Path already exists in PATH.
    )
)

REM Set CUDA_PATH environment variable
echo Setting CUDA_PATH environment variable...
setx CUDA_PATH "!CUDA_PATH[%SELECTED%]!" /M

echo.
echo CUDA paths have been added to the system PATH.
echo Please restart your computer for the changes to take effect.
echo.

pause
exit /b

:CheckPath
REM Check if the path is already in the PATH environment variable
REM Returns 0 if found, 1 if not found
echo %PATH% | findstr /C:"%~1" >nul
if %ERRORLEVEL% EQU 0 (
    exit /b 0
) else (
    exit /b 1
) 