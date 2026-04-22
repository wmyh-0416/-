# Training Environment Report

## Status

- Environment setup: `OK`
- Qwen2.5 smoke test: `OK`
- SFT dry run: `OK`
- DPO dry run: `OK`
- Formal training started: `No`

## Python

- Executable: `/home/ym3447/学霸学渣SFT+DPO/.venv/bin/python`
- Version: `3.9.21`

## CUDA / GPU

- CUDA available in PyTorch: `Yes`
- PyTorch CUDA version: `12.8`
- Driver version: `580.126.09`
- GPU: `NVIDIA A100-SXM4-40GB`
- GPU memory: `40960 MiB`
- Compute capability: `8.0`
- bf16 supported: `Yes`
- Default runtime precision: `bf16`
- Default quantization path: `4-bit QLoRA`

## Base Model

- Hub ID: `Qwen/Qwen2.5-3B-Instruct`
- Local model path: `/home/ym3447/学霸学渣SFT+DPO/models/Qwen2.5-3B-Instruct`

## Key Dependencies

- `torch==2.8.0+cu128`
- `transformers==4.57.6`
- `trl==0.24.0`
- `peft==0.17.1`
- `datasets==4.5.0`
- `accelerate==1.10.1`
- `bitsandbytes==0.48.2`
- `numpy==1.26.4`
- `PyYAML==6.0.2`

## Smoke Test

- Script: `scripts/sanity_check_qwen25.py`
- Config: `config/base.yaml`
- Result JSON: `outputs/reports/qwen25_smoke_test.json`
- Model load: `OK`
- Tokenizer load: `OK`
- Forward pass: `OK`
- Generation: `OK`
- Generated text: `smoke test`
- Forward logits shape: `[1, 28, 151936]`

## Shared Project Files

- Requirements: `requirements_training.txt`
- Shared paths/config: `config/base.yaml`
- Shared helpers: `src/training_utils.py`
- SFT skeleton: `scripts/train_sft_qlora.py`
- DPO skeleton: `scripts/train_dpo_qlora.py`

## Notes

- `transformers` is new enough for `qwen2` model loading and Qwen2.5 chat-template usage.
- The current stack emits an upstream warning that Python `3.9` will be dropped in a future release. The environment is working now, but moving to Python `3.10+` is recommended before long-term training work.
- The SFT skeleton defaults to `assistant_only_loss: false` because the current Qwen chat template does not expose the assistant mask expected by TRL for `assistant_only_loss=True`.
