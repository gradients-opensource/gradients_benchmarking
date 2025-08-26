#!/bin/bash

# Gradients Benchmarking Setup Script
# This script sets up the environment for running evaluations

set -e  # Exit on error

echo "================================================"
echo "Gradients Benchmarking - Environment Setup"
echo "================================================"
echo ""

# Detect Python command - try python3 first, then python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "Error: Python is not installed. Please install Python 3.9 or higher."
    echo ""
    echo "On Debian/Ubuntu:"
    echo "  sudo apt update && sudo apt install python3 python3-pip python3-venv"
    echo ""
    echo "On RHEL/CentOS/Fedora:"
    echo "  sudo yum install python3 python3-pip"
    exit 1
fi

# Check Python version (requires 3.9+)
PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.9"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "Error: Python $REQUIRED_VERSION or higher is required (found $PYTHON_VERSION)"
    exit 1
fi

echo "✓ Python $PYTHON_VERSION detected (using $PYTHON_CMD)"
echo ""

# Check if virtual environment exists and is valid
if [ -d "venv" ]; then
    if [ ! -f "venv/bin/activate" ]; then
        echo "⚠️  Virtual environment appears corrupted. Removing and recreating..."
        rm -rf venv
    else
        echo "✓ Virtual environment already exists"
    fi
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    if ! $PYTHON_CMD -m venv venv 2>/dev/null; then
        echo ""
        echo "❌ Failed to create virtual environment"
        echo ""
        echo "Installing python3-venv is required. Running:"
        echo "  sudo apt update && sudo apt install python3-venv"
        echo ""
        # Try to install it automatically if sudo is available
        if command -v sudo &> /dev/null; then
            read -p "Would you like to install it now? (y/n) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                sudo apt update && sudo apt install -y python3-venv
                echo "Retrying virtual environment creation..."
                $PYTHON_CMD -m venv venv
            else
                echo "Please install python3-venv manually and run this script again."
                exit 1
            fi
        else
            echo "Please install python3-venv manually and run this script again."
            exit 1
        fi
    fi
    echo "✓ Virtual environment created"
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