@echo off
REM ========================================
REM Ceramatic 2.0 - Windows Installation & Launch Script
REM ========================================

echo.
echo =========================================
echo   Ceramatic 2.0 - Setup Windows
echo =========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERRORE] Python non trovato! 
    echo.
    echo Scarica Python da: https://www.python.org/downloads/
    echo Assicurati di selezionare "Add Python to PATH" durante l'installazione
    echo.
    pause
    exit /b 1
)

echo [OK] Python trovato
python --version
echo.

REM Check if virtual environment exists
if not exist ".venv" (
    echo [INFO] Creazione ambiente virtuale...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERRORE] Impossibile creare ambiente virtuale
        echo Prova: python -m pip install --user virtualenv
        pause
        exit /b 1
    )
    echo [OK] Ambiente virtuale creato
) else (
    echo [OK] Ambiente virtuale esistente trovato
)

echo.
echo [INFO] Attivazione ambiente virtuale...
call .venv\Scripts\activate.bat

echo.
echo [INFO] Aggiornamento pip...
python -m pip install --upgrade pip --quiet

echo.
echo [INFO] Installazione dipendenze...
echo Questo potrebbe richiedere alcuni minuti...
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo [ERRORE] Installazione dipendenze fallita
    echo Prova a eseguire manualmente:
    echo   pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo [OK] Dipendenze installate con successo!
echo.

REM Check if model exists
if not exist "Ceramatic_model_V1.pt" (
    echo ==========================================
    echo   ATTENZIONE: Modello non trovato!
    echo ==========================================
    echo.
    echo Il file del modello "Ceramatic_model_V1.pt" non e' presente.
    echo.
    echo Scaricalo da:
    echo https://drive.google.com/file/d/1b23yWPZ0LKerIM8CWz2DcbhapThCnT7A/view
    echo.
    echo e posizionalo in questa cartella.
    echo ==========================================
    echo.
)

echo =========================================
echo   Avvio Ceramatic 2.0 Gradio Interface
echo =========================================
echo.
echo L'applicazione si aprira' nel browser...
echo Per chiudere: premi Ctrl+C in questa finestra
echo.

REM Launch the Gradio app
python ceramatic_gradio.py

if errorlevel 1 (
    echo.
    echo [ERRORE] Impossibile avviare l'applicazione
    pause
    exit /b 1
)

echo.
echo Applicazione terminata.
pause