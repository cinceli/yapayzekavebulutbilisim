@echo off
REM Ilk kurulum: .venv olusturur ve requirements.txt paketlerini yukler.
REM Mevcut bir .venv varsa siler ve yeniden kurar.

cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [HATA] Python bulunamadi. https://www.python.org/downloads/windows/ adresinden
    echo        Python 3.12 kurun ve kurulum sirasinda "Add python.exe to PATH" kutusunu isaretleyin.
    pause
    exit /b 1
)

if exist ".venv" (
    echo Eski .venv siliniyor...
    rmdir /s /q ".venv"
)

echo Sanal ortam olusturuluyor...
python -m venv .venv
if errorlevel 1 (
    echo [HATA] venv olusturulamadi.
    pause
    exit /b 1
)

echo pip guncelleniyor...
".venv\Scripts\python.exe" -m pip install --upgrade pip

echo Paketler yukleniyor (1-3 dakika)...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [HATA] Paket kurulumu basarisiz.
    pause
    exit /b 1
)

echo.
echo [TAMAM] Kurulum bitti. Simdi:
echo   - egit.bat  : modeli egit
echo   - test.bat  : testleri calistir
echo   - arayuz.bat: web arayuzunu ac

pause
