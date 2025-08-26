# Reproduce Gradients Instruct Benchmark Evaluations

## Quick Start

If you haven't already cloned this repository:
```bash
git clone -b dev https://github.com/gradients-opensource/gradients_benchmarking.git
cd gradients_benchmarking
```

> **Note:** Alternatively, you can clone the official repo `https://github.com/EleutherAI/lm-evaluation-harness.git` and set `num_fewshot: 0` in `gsm8k.yaml`

## Setup

Run the setup script to install all dependencies:
```bash
./setup.sh
source venv/bin/activate
```

## Running Batch Evaluations

Evaluate multiple models at once:
```bash
python batch_evaluate.py --config configs/batch_eval_config.yaml
```

This will evaluate all models specified in the config file and generate:
- Individual results for each model
- Consolidated JSON results  
- Summary CSV for easy comparison

Results will be saved to `output/batch_eval/` with timestamps.
