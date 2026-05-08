@echo off
REM Streamlit arayuzunu baslatir. Tarayici otomatik acilir (http://localhost:8501).
REM Durdurmak icin terminale donup Ctrl+C bas.

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [HATA] .venv bulunamadi. Once kur.bat dosyasini calistir.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m streamlit run app.py

pause
