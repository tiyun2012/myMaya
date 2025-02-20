
@echo off
REM Execute the PowerShell script located in the same directory as the batch file
powershell.exe -File "%~dp0run_export.ps1"
