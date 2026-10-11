@echo off
REM Customer Success AI Copilot - laptop demo launcher.
REM Double-click this file, or run it from a terminal in the project folder.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_demo.ps1"
pause
