#!/bin/bash

# Gradients Benchmarking Setup Script
# This script sets up the environment for running evaluations

set -e  # Exit on error

echo "================================================"
echo "Gradients Benchmarking - Environment Setup"
echo "================================================"
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed. Please install Python 3.9 or higher."
    exit 1
fi

# Check Python version (requires 3.9+)
PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.9"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "Error: Python $REQUIRED_VERSION or higher is required (found $PYTHON_VERSION)"
    exit 1
fi

echo "✓ Python $PYTHON_VERSION detected"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --quiet --upgrade pip

# Install the package with required extras
echo ""
echo "Installing lm-evaluation-harness with required dependencies..."
echo "This may take a few minutes..."

# Install base package with math, ifeval, and sentencepiece extras for leaderboard
pip install -e ".[math,ifeval,sentencepiece]"

# Install PyYAML for batch evaluation config
pip install PyYAML>=6.0

# Optional: Install hf_transfer for faster model downloads
pip install hf_transfer

echo ""
echo "✓ All dependencies installed"

# Create output directory
mkdir -p output/batch_eval
echo "✓ Created output directory: output/batch_eval"

echo ""
echo "================================================"
echo "✅ Setup Complete!"
echo "================================================"
echo ""
echo "To activate the environment in future sessions:"
echo "  source venv/bin/activate"
echo ""
echo "To run batch evaluation:"
echo "  python batch_evaluate.py --config configs/batch_eval_config.yaml"
echo ""
echo "To run individual model evaluation:"
echo "  export HF_ALLOW_CODE_EVAL=1"
echo "  lm_eval --model hf --model_args pretrained=<MODEL_ID> --tasks leaderboard,gsm8k --device cuda --batch_size auto"
echo ""