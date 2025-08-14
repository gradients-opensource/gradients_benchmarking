# Reproduce Gradients Instruct Benchmark Evaluations

## Step 1: Clone

```bash
git clone -b dev https://github.com/samoline1/lm-evaluation-harness.git
```

> **Note:** Alternatively, you can clone the official repo `https://github.com/EleutherAI/lm-evaluation-harness.git` and set `num_fewshot: 0` in `gsm8k.yaml`

## Step 2: Prepare Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e ".[math,ifeval,sentencepiece]"  # for leaderboard
pip install python-dotenv
pip install wandb
```

## Step 3: Logging to wandb and Hugging Face

```bash
wandb login
huggingface-cli login  # you will need to accept conditions of a gated HF dataset repo
```

## Step 4: Evaluate Models on Benchmarks

Replace `<MODEL_ID>` with one of the following models:
- `Qwen/Qwen3-8B-Base`
- `Qwen/Qwen3-8B`
- `samoline/e9729fda-9a6b-44ee-a717-7afdc47f0da8`

Then run:
```bash
export HF_ALLOW_CODE_EVAL=1 && \
lm_eval \
  --model hf \
  --model_args pretrained=<MODEL_ID> \
  --tasks leaderboard,gsm8k \
  --device cuda \
  --batch_size auto \
  --output_path output/gradients \
  --wandb_args project=gradients-evaluation,name=<MODEL_ID> \
  --confirm_run_unsafe_code \
  --log_samples
```
