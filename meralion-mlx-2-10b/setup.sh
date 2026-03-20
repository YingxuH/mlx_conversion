#!/usr/bin/env bash
# MERaLiON-MLX setup script
# Creates a virtual environment and installs all dependencies.
# Usage: chmod +x setup.sh && ./setup.sh

set -euo pipefail

# --- Check macOS Apple Silicon ---
if [ "$(uname -s)" != "Darwin" ]; then
    echo "ERROR: This project requires macOS (detected: $(uname -s))."
    exit 1
fi

ARCH=$(uname -m)
if [ "$ARCH" != "arm64" ]; then
    echo "ERROR: Apple Silicon (arm64) required. Detected: $ARCH"
    echo "MLX does not support Intel Macs."
    exit 1
fi

# --- Find Python 3.10+ ---
PYTHON=""
for candidate in python3.14 python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" -c "import sys; print(sys.version_info[:2])" 2>/dev/null) || continue
        major=$(echo "$ver" | tr -d '(),' | awk '{print $1}')
        minor=$(echo "$ver" | tr -d '(),' | awk '{print $2}')
        if [ "$major" -eq 3 ] && [ "$minor" -ge 10 ]; then
            PYTHON=$(command -v "$candidate")
            echo "Found Python $major.$minor at $PYTHON"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERROR: Python 3.10+ not found."
    echo "Install via: brew install python@3.12"
    exit 1
fi

# --- Create virtual environment ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
    echo "Delete it first if you want a fresh install: rm -rf $VENV_DIR"
    exit 1
fi

echo "Creating virtual environment..."
"$PYTHON" -m venv "$VENV_DIR"

# --- Install dependencies ---
echo "Installing dependencies (this may take a few minutes)..."
"$VENV_DIR/bin/pip" install --upgrade pip --quiet
"$VENV_DIR/bin/pip" install -e "$SCRIPT_DIR"

echo ""
echo "================================================"
echo "  Setup complete!"
echo "================================================"
echo ""
echo "Activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "Run inference:"
echo "  python scripts/inference.py \\"
echo "    --model-dir models/2-3b-mlx \\"
echo "    --audio YOUR_AUDIO.wav \\"
echo "    --task asr"
echo ""
