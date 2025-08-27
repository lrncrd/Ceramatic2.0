#!/bin/bash

# ========================================
# Ceramatic 2.0 - Fast Startup Script
# ========================================

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}  Ceramatic 2.0 - Quick Start${NC}"
echo -e "${BLUE}=========================================${NC}"

# Detect Python command
if command -v python3 &>/dev/null; then
    PYTHON_CMD=python3
elif command -v python &>/dev/null; then
    PYTHON_CMD=python
else
    echo -e "${RED}[ERRORE]${NC} Python non trovato!"
    exit 1
fi

# Quick check for virtual environment
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}[INFO]${NC} Creazione ambiente virtuale..."
    $PYTHON_CMD -m venv .venv
fi

# Activate virtual environment
echo -e "${GREEN}[INFO]${NC} Attivazione ambiente..."
source .venv/bin/activate

# Check if dependencies are already installed (fast check)
if ! pip show gradio &>/dev/null 2>&1; then
    echo -e "${YELLOW}[INFO]${NC} Prima installazione - installazione dipendenze..."
    
    # macOS specific optimizations
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if [[ $(uname -m) == "arm64" ]]; then
            export ARCHFLAGS="-arch arm64"
        fi
        # Check for OpenBLAS only if scipy not installed
        if ! pip show scipy &>/dev/null 2>&1 && command -v brew &>/dev/null; then
            if ! brew list openblas &>/dev/null 2>&1; then
                echo -e "${YELLOW}[INFO]${NC} Installazione OpenBLAS..."
                brew install openblas &>/dev/null 2>&1
            fi
            export OPENBLAS="$(brew --prefix openblas 2>/dev/null)"
        fi
    fi
    
    # Install dependencies quietly
    pip install --upgrade pip wheel setuptools --quiet
    echo -e "${YELLOW}[INFO]${NC} Installazione pacchetti (questo richiederà qualche minuto solo la prima volta)..."
    pip install -r requirements.txt --quiet --disable-pip-version-check
    echo -e "${GREEN}[OK]${NC} Dipendenze installate!"
else
    echo -e "${GREEN}[OK]${NC} Dipendenze già installate"
fi

# Check for model in multiple locations
MODEL_FOUND=false
for path in "Ceramatic_model_V1.pt" "models/Ceramatic_model_V1.pt" "model/Ceramatic_model_V1.pt"; do
    if [ -f "$path" ]; then
        echo -e "${GREEN}[OK]${NC} Modello trovato: $path"
        MODEL_FOUND=true
        break
    fi
done

if [ "$MODEL_FOUND" = false ]; then
    echo ""
    echo -e "${YELLOW}=========================================${NC}"
    echo -e "${YELLOW}  ⚠️  Modello non trovato${NC}"
    echo -e "${YELLOW}=========================================${NC}"
    echo "Scarica da: https://drive.google.com/file/d/1b23yWPZ0LKerIM8CWz2DcbhapThCnT7A/view"
    echo "Posiziona in: ./models/ o ./"
    echo ""
fi

# Detect GPU
if [[ "$OSTYPE" == "darwin"* ]] && [[ $(uname -m) == "arm64" ]]; then
    echo -e "${GREEN}[INFO]${NC} Apple Silicon (MPS) disponibile"
elif command -v nvidia-smi &>/dev/null 2>&1; then
    echo -e "${GREEN}[INFO]${NC} GPU NVIDIA disponibile"
else
    echo -e "${YELLOW}[INFO]${NC} Uso CPU"
fi

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}  Avvio Ceramatic 2.0${NC}"
echo -e "${BLUE}=========================================${NC}"

# Launch the Gradio app
exec python ceramatic_gradio.py