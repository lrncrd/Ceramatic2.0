#!/bin/bash

# Script per liberare le porte usate da Ceramatic

echo "🔍 Ricerca processi su porte 7860-7870..."

# Kill all Python processes related to ceramatic or gradio
pkill -f "python.*ceramatic" 2>/dev/null
pkill -f "python.*gradio" 2>/dev/null

# Try to kill processes on specific ports
for port in {7860..7870}; do
    # macOS specific command
    pid=$(lsof -t -i:$port 2>/dev/null)
    if [ ! -z "$pid" ]; then
        echo "Killing process $pid on port $port"
        kill -9 $pid 2>/dev/null
    fi
done

echo "✅ Porte liberate!"
echo ""
echo "Ora puoi avviare Ceramatic con:"
echo "./install_and_run.sh"