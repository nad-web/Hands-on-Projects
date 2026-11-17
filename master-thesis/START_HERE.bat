@echo off
cd /d "%~dp0"
py -3 tools\serve.py
if errorlevel 1 pause
