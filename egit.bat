@echo off
REM Demo veriyle hizli egitim yapar. Metrikleri konsola yazar,
REM modeli models\pipeline_artifacts.joblib dosyasina kaydeder.

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [HATA] .venv bulunamadi. Once kur.bat dosyasini calistir.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" train.py --demo

pause
