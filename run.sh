#!/usr/bin/env bash

echo ""
echo " =============================="
echo "  MirrorMetrics - Starting..."
echo " =============================="
echo ""

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate the virtual environment
source "$SCRIPT_DIR/venv/bin/activate"

# Run the Python script
python3 "$SCRIPT_DIR/mirror_metrics.py"

echo ""
read -rsp $'Press any key to continue...\n' -n1 key
