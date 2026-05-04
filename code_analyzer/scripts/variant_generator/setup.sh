#!/bin/bash

# Quick Setup & Execution Guide for Code Variant Generation Pipeline

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║          Code Variant Generation Pipeline - Quick Setup Guide             ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo

# Check if Ollama is running
echo "[1/4] Checking Ollama setup..."
if command -v ollama &> /dev/null; then
    echo "  ✓ Ollama is installed"
else
    echo "  ✗ Ollama not found. Please install from: https://ollama.ai"
    exit 1
fi

# Check Python
echo
echo "[2/4] Checking Python..."
if python3 --version &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "  ✓ $PYTHON_VERSION available"
else
    echo "  ✗ Python 3 not found"
    exit 1
fi

# Check dependencies
echo
echo "[3/4] Checking Python dependencies..."
if python3 -c "import pandas, ollama" 2>/dev/null; then
    echo "  ✓ pandas and ollama packages installed"
else
    echo "  ⚠ Installing required packages: pandas, ollama"
    pip install pandas ollama
fi

# Setup directory structure
echo
echo "[4/4] Setting up directory structure..."
mkdir -p "$(dirname "$0")/../output/logs"
echo "  ✓ Output directories ready"

echo
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                            Ready to Run!                                  ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo
echo "To start the full pipeline, run:"
echo
echo "    bash $(dirname "$0")/run_pipeline.sh"
echo
echo "For individual dataset processing:"
echo
echo "    # Human Eval with specific model"
echo "    python3 $(dirname "$0")/generate_human_eval_variants.py qwen2.5-coder:7b"
echo
echo "    # Class Eval with specific model"
echo "    python3 $(dirname "$0")/generate_class_eval_variants.py llama2:8b"
echo
echo "Models available:"
echo "  - qwen2.5-coder:7b"
echo "  - llama3.1:8b"
echo "  - deepseek-r1:8b"
echo
