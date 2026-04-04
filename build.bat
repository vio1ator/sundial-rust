@echo off
REM Build script for Sundial Rust on Windows
REM Usage: build.bat [OPTIONS]
REM
REM Options:
REM   --target <TRIPLE>    Target platform (e.g., x86_64-pc-windows-msvc)
REM   --profile <PROFILE>  Build profile (default: release)
REM   --clean              Clean build artifacts before building
REM   --help               Show this help message
REM
REM Examples:
REM   build.bat                          # Build for current platform
REM   build.bat --target x86_64-pc-windows-msvc
REM   build.bat --profile release --clean

setlocal enabledelayedexpansion

REM Default values
set TARGET=
set PROFILE=release
set CLEAN=false

REM Parse arguments
:parse_args
if "%~1"=="" goto parse_end
if "%~1"=="--target" (
    set TARGET=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="--profile" (
    set PROFILE=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="--clean" (
    set CLEAN=true
    shift
    goto parse_args
)
if "%~1"=="--help" (
    call :show_help
    goto end
)
echo Error: Unknown option: %~1
echo Use --help for usage information
goto end
:parse_end

REM Validate profile
findstr /c:"[profile.%PROFILE%]" Cargo.toml >nul 2>&1
if errorlevel 1 (
    echo Warning: Profile '%PROFILE%' not found in Cargo.toml, using default release settings
)

REM Clean if requested
if "%CLEAN%"=="true" (
    echo Cleaning build artifacts...
    cargo clean
    if errorlevel 1 (
        echo Error: Failed to clean build artifacts
        goto end
    )
)

REM Add target if specified
if not "%TARGET%"=="" (
    echo Adding Rust target: %TARGET%
    rustup target add %TARGET%
    if errorlevel 1 (
        echo Error: Failed to add target: %TARGET%
        goto end
    )
)

REM Build command
set BUILD_CMD=cargo build

if not "%TARGET%"=="" (
    set BUILD_CMD=%BUILD_CMD% --target %TARGET%
)

set BUILD_CMD=%BUILD_CMD% --profile %PROFILE%

echo Running: %BUILD_CMD%
echo.

REM Execute build
%BUILD_CMD%

REM Check if build succeeded
if errorlevel 1 (
    echo Error: Build failed
    goto end
)

echo.
echo Build successful!

REM Show binary location
if not "%TARGET%"=="" (
    set BINARY_PATH=target\%TARGET%\%PROFILE%\sundial-rust.exe
) else (
    set BINARY_PATH=target\%PROFILE%\sundial-rust.exe
)

if exist %BINARY_PATH% (
    echo Binary location: %BINARY_PATH%
)

goto end

:show_help
echo Build script for Sundial Rust on Windows
echo.
echo Usage: build.bat [OPTIONS]
echo.
echo Options:
echo   --target ^<TRIPLE^>    Target platform ^(e.g., x86_64-pc-windows-msvc^)
echo   --profile ^<PROFILE^>  Build profile ^(default: release^)
echo   --clean              Clean build artifacts before building
echo   --help               Show this help message
echo.
echo Examples:
echo   build.bat                          ^# Build for current platform
echo   build.bat --target x86_64-pc-windows-msvc
echo   build.bat --profile release --clean
echo.
echo Profiles available:
echo   release                    Standard release build
echo   release-x86_64-linux       Linux x86_64 with full LTO
echo   release-aarch64-linux      Linux ARM64 with Thin LTO
echo   release-x86_64-windows     Windows x86_64 with MSVC
echo   release-aarch64-macos      macOS ARM64
echo   release-x86_64-macos       macOS x86_64
goto :eof

:end
endlocal
