@echo off
REM Tum birim testleri (pytest) calistirir.

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [HATA] .venv bulunamadi. Once kur.bat dosyasini calistir.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m pytest tests/ -q

pause
