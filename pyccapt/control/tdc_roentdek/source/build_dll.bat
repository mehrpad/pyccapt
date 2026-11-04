@echo off
rem Builds ..\wrapper_read_TDC8HP_x64.dll (Release, static CRT, no MFC) with the
rem VS 2019/2022 v142 toolset. Run from any prompt; needs "MSVC v142 x64" installed.
set VCVARS=
for /f "usebackq delims=" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -find VC\Auxiliary\Build\vcvars64.bat`) do set VCVARS=%%i
if not defined VCVARS (echo Visual Studio C++ build tools not found & exit /b 1)
call "%VCVARS%" -vcvars_ver=14.29 >nul || exit /b 1
cd /d "%~dp0"
set OUT=%TEMP%\pyccapt_tdc8hp_build
if not exist "%OUT%" mkdir "%OUT%"
cl /nologo /LD /EHsc /O2 /MT /W3 /DWIN64 /D_CRT_SECURE_NO_WARNINGS /D_MBCS wrapper_read_TDC8HP.cpp LMF_IO.cpp /Fo"%OUT%\\" /Fe"%OUT%\wrapper_read_TDC8HP_x64.dll" /link hptdc_driver_3.5.2_x64.lib || exit /b 1
copy /y "%OUT%\wrapper_read_TDC8HP_x64.dll" "..\wrapper_read_TDC8HP_x64.dll" >nul || exit /b 1
echo Built ..\wrapper_read_TDC8HP_x64.dll
