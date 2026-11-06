@echo off
rem ---------------------------------------------------------------------------
rem Builds the RoentDek wrapper DLL (x64, Release, static C runtime) and tools.
rem Needs Visual Studio 2022 (C++ workload). No MFC needed.
rem
rem   ..\wrapper_read_TDC8HP_x64.dll        acquisition DLL (needs hptdc_driver_3.5.2_x64.dll at run time)
rem   ..\roentdek_resort_x64.dll            RoentDek Resort64c reconstruction (needs resort64c_VS2010_x64.dll
rem                                         and the VC++ 2010 runtime, both also used by CoboldPC)
rem   ..\tools\lmf_dump.exe                 LMF to binary dump (offline evaluation)
rem   test\wrapper_read_TDC8HP_x64_mock.dll test DLL that replays a recorded run (no hardware)
rem   test\resort_replay.exe                LMF replay through Resort64c (test tool)
rem
rem The two Resort64c steps need RoentDek's resort64c.h and resort64c_VS2010_x64.lib (from the
rem CoboldPC installation) in source\resort64c. They are skipped if these files are missing.
rem ---------------------------------------------------------------------------
setlocal
set VCVARS="C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist %VCVARS% set VCVARS="C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
if not exist %VCVARS% set VCVARS="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
call %VCVARS% >nul 2>nul
cd /d %~dp0
if not exist obj mkdir obj
if not exist obj\mock mkdir obj\mock
if not exist test mkdir test
if not exist ..\tools mkdir ..\tools
set CFLAGS=/nologo /EHsc /O2 /MT /W3 /D_CRT_SECURE_NO_WARNINGS /DWIN64 /DWIN32 /DNDEBUG

echo === acquisition DLL
cl %CFLAGS% /LD /Foobj\ wrapper_read_TDC8HP.cpp LMF_IO.cpp /Fe:..\wrapper_read_TDC8HP_x64.dll /link hptdc_driver_3.5.2_x64.lib || goto :error

echo === mock DLL, test only
cl %CFLAGS% /DMOCK_TDC /LD /Foobj\mock\ wrapper_read_TDC8HP.cpp LMF_IO.cpp /Fe:test\wrapper_read_TDC8HP_x64_mock.dll || goto :error

if not exist resort64c\resort64c.h goto :no_resort
if not exist resort64c\resort64c_VS2010_x64.lib goto :no_resort
echo === Resort64c reconstruction DLL
cl %CFLAGS% /LD /Iresort64c /Foobj\ resort_wrapper.cpp /Fe:..\roentdek_resort_x64.dll /link /LIBPATH:resort64c resort64c_VS2010_x64.lib || goto :error
if exist resort64c\resort64c_VS2010_x64.dll copy /Y resort64c\resort64c_VS2010_x64.dll ..\ >nul

echo === resort_replay test harness
cl %CFLAGS% /Iresort64c /Foobj\ resort_replay.cpp LMF_IO.cpp /Fe:test\resort_replay.exe /link /LIBPATH:resort64c resort64c_VS2010_x64.lib || goto :error
goto :lmf_dump

:no_resort
echo === Resort64c skipped: resort64c.h and resort64c_VS2010_x64.lib (from CoboldPC) are not in source\resort64c
echo     the existing roentdek_resort_x64.dll is kept

:lmf_dump

echo === lmf_dump
cl %CFLAGS% /Foobj\ lmf_dump.cpp LMF_IO.cpp /Fe:..\tools\lmf_dump.exe || goto :error

echo BUILD OK
exit /b 0
:error
echo BUILD FAILED
exit /b 1
