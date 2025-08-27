# 📦 Guida Installazione Ceramatic 2.0

## 🚀 Quick Start

### Metodo Universale (Tutti i Sistemi)
```bash
python setup_ceramatic.py
```

## 💻 Installazione per Sistema Operativo

### Windows

#### Opzione 1: Script Batch (CMD)
Doppio click su `install_and_run_windows.bat` o esegui:
```cmd
install_and_run_windows.bat
```

#### Opzione 2: PowerShell (Consigliato)
```powershell
# Se necessario, abilita esecuzione script
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Esegui lo script
.\install_and_run_windows.ps1
```

#### Opzione 3: Python Universale
```cmd
python setup_ceramatic.py
```

### macOS

#### Opzione 1: Script Bash
```bash
chmod +x install_and_run.sh
./install_and_run.sh
```

#### Opzione 2: Python Universale
```bash
python3 setup_ceramatic.py
```

### Linux (Ubuntu/Debian/Fedora/Arch)

#### Opzione 1: Script Bash
```bash
chmod +x install_and_run.sh
./install_and_run.sh
```

#### Opzione 2: Python Universale
```bash
python3 setup_ceramatic.py
```

## 🛠️ Opzioni Avanzate

### Script Python Universale (`setup_ceramatic.py`)

```bash
# Installazione completa e avvio
python setup_ceramatic.py

# Solo installazione (senza avviare)
python setup_ceramatic.py --install-only

# Avvio rapido (salta setup se già installato)
python setup_ceramatic.py --skip-setup

# Specifica porta personalizzata
python setup_ceramatic.py --port 8080

# Crea link pubblico temporaneo
python setup_ceramatic.py --share

# Esponi su rete locale
python setup_ceramatic.py --host 0.0.0.0
```

## 📋 Prerequisiti

### Tutti i Sistemi
- Python 3.7 o superiore
- 8GB RAM (16GB consigliato)
- 5GB spazio libero su disco

### Windows
- Windows 10/11
- Python da [python.org](https://www.python.org/downloads/)
- **IMPORTANTE**: Durante installazione Python, seleziona "Add Python to PATH"

### macOS
- macOS 10.14+
- Python via Homebrew: `brew install python@3.9`
- O download da [python.org](https://www.python.org/downloads/)
- **Per scipy**: OpenBLAS via Homebrew: `brew install openblas`

### Linux
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3 python3-pip python3-venv

# Fedora
sudo dnf install python3 python3-pip

# Arch
sudo pacman -S python python-pip
```

## 🔧 Troubleshooting

### Errore: "Python non trovato"
- **Windows**: Reinstalla Python con "Add to PATH" abilitato
- **macOS**: Installa con `brew install python@3.9`
- **Linux**: `sudo apt-get install python3`

### Errore: "pip non trovato"
```bash
# Windows
python -m ensurepip --upgrade

# macOS/Linux
python3 -m ensurepip --upgrade
```

### Errore: "Modulo venv non disponibile"
```bash
# Ubuntu/Debian
sudo apt-get install python3-venv

# Altri sistemi
pip install virtualenv
```

### GPU non rilevata
- **NVIDIA**: Installa CUDA Toolkit e driver aggiornati
- **Apple Silicon**: Automaticamente rilevato su macOS 12+
- **AMD**: Supporto limitato, usa CPU

## 📥 Download Modello

Il modello YOLO pre-addestrato è richiesto per l'elaborazione:

1. Scarica da: [Google Drive](https://drive.google.com/file/d/1b23yWPZ0LKerIM8CWz2DcbhapThCnT7A/view)
2. Salva come `Ceramatic_model_V1.pt` nella cartella del progetto
3. Dimensione: ~130MB

## 🌐 Accesso Remoto

Per accedere all'app da altri dispositivi sulla rete:

```bash
# Esponi su tutte le interfacce di rete
python setup_ceramatic.py --host 0.0.0.0 --port 7860
```

Poi accedi da browser: `http://[IP-DEL-COMPUTER]:7860`

## 🐳 Docker (Opzionale)

Coming soon: Dockerfile per deployment containerizzato

## ❓ Supporto

- **Issues**: [GitHub Repository](https://github.com/lrncrd/Ceramatic2.0/issues)
- **Documentazione**: README.md
- **Paper**: [DOI: 10.1016/j.daach.2025.e00435](https://doi.org/10.1016/j.daach.2025.e00435)

## ✅ Verifica Installazione

Dopo l'installazione, verifica che tutto funzioni:

```bash
# Attiva ambiente virtuale
# Windows
.\.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

# Verifica dipendenze
pip list | grep gradio
pip list | grep ultralytics

# Test import
python -c "import gradio; import ultralytics; print('OK')"
```

Se vedi "OK", l'installazione è riuscita!