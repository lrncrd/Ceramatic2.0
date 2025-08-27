# ========================================
# Ceramatic 2.0 - Windows PowerShell Installation & Launch Script
# ========================================
# Run with: powershell -ExecutionPolicy Bypass -File install_and_run_windows.ps1

$ErrorActionPreference = "Stop"

# Colors for output
$Host.UI.RawUI.ForegroundColor = "White"

function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    $previousColor = $Host.UI.RawUI.ForegroundColor
    $Host.UI.RawUI.ForegroundColor = $Color
    Write-Host $Message
    $Host.UI.RawUI.ForegroundColor = $previousColor
}

function Test-Administrator {
    $currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
    return $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

Write-Host ""
Write-ColorOutput "=========================================" "Cyan"
Write-ColorOutput "  Ceramatic 2.0 - Setup Windows" "Cyan"
Write-ColorOutput "=========================================" "Cyan"
Write-Host ""

# Check if running as administrator (optional, for system-wide installs)
if (Test-Administrator) {
    Write-ColorOutput "[INFO] Esecuzione con privilegi di amministratore" "Yellow"
}

# Check Python installation
Write-ColorOutput "[INFO] Controllo installazione Python..." "Cyan"

$pythonCmd = $null
$pythonVersion = $null

# Try multiple Python commands
$pythonCommands = @("python", "python3", "py")

foreach ($cmd in $pythonCommands) {
    try {
        $version = & $cmd --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            $pythonCmd = $cmd
            $pythonVersion = $version
            break
        }
    } catch {
        continue
    }
}

if ($null -eq $pythonCmd) {
    Write-ColorOutput "[ERRORE] Python non trovato!" "Red"
    Write-Host ""
    Write-Host "Scarica Python da: https://www.python.org/downloads/"
    Write-Host "IMPORTANTE: Durante l'installazione, seleziona 'Add Python to PATH'"
    Write-Host ""
    Write-Host "Premi un tasto per uscire..."
    $host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") | Out-Null
    exit 1
}

Write-ColorOutput "[OK] Python trovato: $pythonVersion" "Green"

# Get Python version details
$versionInfo = & $pythonCmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>&1
$majorMinor = [double]$versionInfo

if ($majorMinor -lt 3.7) {
    Write-ColorOutput "[ERRORE] Python $versionInfo trovato, ma richiesto 3.7+" "Red"
    exit 1
}

# Check for venv module
Write-ColorOutput "[INFO] Controllo modulo venv..." "Cyan"
& $pythonCmd -m venv --help 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-ColorOutput "[ERRORE] Modulo venv non disponibile" "Red"
    Write-Host "Installa con: $pythonCmd -m pip install virtualenv"
    exit 1
}

# Create virtual environment
if (-not (Test-Path ".venv")) {
    Write-ColorOutput "[INFO] Creazione ambiente virtuale..." "Cyan"
    & $pythonCmd -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput "[ERRORE] Impossibile creare ambiente virtuale" "Red"
        exit 1
    }
    Write-ColorOutput "[OK] Ambiente virtuale creato" "Green"
} else {
    Write-ColorOutput "[OK] Ambiente virtuale esistente trovato" "Green"
}

# Activate virtual environment
Write-Host ""
Write-ColorOutput "[INFO] Attivazione ambiente virtuale..." "Cyan"

$venvPython = ".\.venv\Scripts\python.exe"
$venvPip = ".\.venv\Scripts\pip.exe"

if (-not (Test-Path $venvPython)) {
    Write-ColorOutput "[ERRORE] Python non trovato nell'ambiente virtuale" "Red"
    exit 1
}

# Upgrade pip
Write-ColorOutput "[INFO] Aggiornamento pip..." "Cyan"
& $venvPython -m pip install --upgrade pip --quiet

if ($LASTEXITCODE -ne 0) {
    Write-ColorOutput "[WARN] Aggiornamento pip non riuscito, continuo..." "Yellow"
}

# Install dependencies
Write-Host ""
Write-ColorOutput "[INFO] Installazione dipendenze..." "Cyan"
Write-Host "Questo potrebbe richiedere alcuni minuti..."

# Check if requirements.txt exists
if (-not (Test-Path "requirements.txt")) {
    Write-ColorOutput "[ERRORE] File requirements.txt non trovato" "Red"
    exit 1
}

# Install packages
& $venvPip install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-ColorOutput "[ERRORE] Installazione dipendenze fallita" "Red"
    Write-Host "Prova a eseguire manualmente:"
    Write-Host "  .\.venv\Scripts\activate"
    Write-Host "  pip install -r requirements.txt"
    Write-Host ""
    Write-Host "Premi un tasto per uscire..."
    $host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") | Out-Null
    exit 1
}

Write-Host ""
Write-ColorOutput "[OK] Dipendenze installate con successo!" "Green"
Write-Host ""

# Check if model exists
if (-not (Test-Path "Ceramatic_model_V1.pt")) {
    Write-ColorOutput "=========================================" "Yellow"
    Write-ColorOutput "  ATTENZIONE: Modello non trovato!" "Yellow"
    Write-ColorOutput "=========================================" "Yellow"
    Write-Host ""
    Write-Host "Il file del modello 'Ceramatic_model_V1.pt' non è presente."
    Write-Host ""
    Write-Host "Scaricalo da:"
    Write-ColorOutput "https://drive.google.com/file/d/1b23yWPZ0LKerIM8CWz2DcbhapThCnT7A/view" "Cyan"
    Write-Host ""
    Write-Host "e posizionalo in questa cartella."
    Write-ColorOutput "=========================================" "Yellow"
    Write-Host ""
}

# Check GPU availability (CUDA, MPS, or CPU)
Write-ColorOutput "[INFO] Controllo disponibilità GPU..." "Cyan"
$deviceCheck = & $venvPython -c @"
import torch
if torch.cuda.is_available():
    print(f'CUDA|{torch.cuda.get_device_name(0)}')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('MPS|Apple Silicon GPU')
else:
    print('CPU|No GPU available')
"@ 2>&1

if ($deviceCheck -like "*CUDA*") {
    $deviceInfo = $deviceCheck.Split('|')[1]
    Write-ColorOutput "[OK] GPU CUDA disponibile: $deviceInfo" "Green"
} elseif ($deviceCheck -like "*MPS*") {
    Write-ColorOutput "[OK] Apple Silicon GPU (MPS) disponibile" "Green"
} else {
    Write-ColorOutput "[INFO] Nessuna GPU rilevata, verrà usata la CPU" "Yellow"
}

Write-Host ""
Write-ColorOutput "=========================================" "Cyan"
Write-ColorOutput "  Avvio Ceramatic 2.0 Gradio Interface" "Cyan"
Write-ColorOutput "=========================================" "Cyan"
Write-Host ""
Write-Host "L'applicazione si aprirà nel browser..."
Write-ColorOutput "Per chiudere: premi Ctrl+C in questa finestra" "Yellow"
Write-Host ""

# Check if Gradio app exists
if (-not (Test-Path "ceramatic_gradio.py")) {
    Write-ColorOutput "[ERRORE] File ceramatic_gradio.py non trovato" "Red"
    exit 1
}

# Launch the Gradio app
try {
    & $venvPython ceramatic_gradio.py
} catch {
    Write-Host ""
    Write-ColorOutput "[ERRORE] Impossibile avviare l'applicazione" "Red"
    Write-Host $_.Exception.Message
    Write-Host ""
    Write-Host "Premi un tasto per uscire..."
    $host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") | Out-Null
    exit 1
}

Write-Host ""
Write-ColorOutput "Applicazione terminata." "Green"
Write-Host "Premi un tasto per uscire..."
$host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") | Out-Null