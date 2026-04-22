from __future__ import annotations

import importlib
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "base.yaml"


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def ensure_dir(path: str | Path) -> Path:
    resolved = resolve_project_path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def resolve_model_source(config: dict[str, Any]) -> str:
    local_dir = resolve_project_path(config["model"]["local_dir"])
    if config["model"].get("prefer_local", True) and local_dir.exists():
        return str(local_dir)
    return str(config["model"]["hub_id"])


def detect_compute_dtype(config: dict[str, Any]) -> tuple[torch.dtype, str]:
    runtime_cfg = config.get("runtime", {})
    if not torch.cuda.is_available():
        return torch.float32, "fp32"
    if runtime_cfg.get("prefer_bf16", True) and torch.cuda.is_bf16_supported():
        return torch.bfloat16, "bf16"
    if runtime_cfg.get("fallback_fp16", True):
        return torch.float16, "fp16"
    return torch.float32, "fp32"


def build_bnb_config(
    config: dict[str, Any],
    compute_dtype: torch.dtype,
    enable_4bit: bool | None = None,
) -> BitsAndBytesConfig | None:
    quant_cfg = config.get("quantization", {})
    use_4bit = quant_cfg.get("enable_4bit", True) if enable_4bit is None else enable_4bit
    if not use_4bit or not torch.cuda.is_available():
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
    )


def build_model_init_kwargs(
    config: dict[str, Any],
    compute_dtype: torch.dtype,
    enable_4bit: bool | None = None,
) -> tuple[dict[str, Any], bool]:
    kwargs: dict[str, Any] = {
        "trust_remote_code": config.get("runtime", {}).get("trust_remote_code", False),
        "low_cpu_mem_usage": config.get("runtime", {}).get("low_cpu_mem_usage", True),
    }
    attn_implementation = config.get("model", {}).get("attn_implementation")
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    quantization_config = build_bnb_config(config, compute_dtype, enable_4bit=enable_4bit)
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
        kwargs["device_map"] = "auto"
        kwargs["dtype"] = compute_dtype
        return kwargs, True
    if torch.cuda.is_available():
        kwargs["dtype"] = compute_dtype
        kwargs["device_map"] = "auto"
    return kwargs, False


def load_tokenizer(model_source: str, config: dict[str, Any]):
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        use_fast=config.get("runtime", {}).get("use_fast_tokenizer", True),
        trust_remote_code=config.get("runtime", {}).get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model_and_tokenizer(
    config: dict[str, Any],
    enable_4bit: bool | None = None,
):
    model_source = resolve_model_source(config)
    compute_dtype, precision_name = detect_compute_dtype(config)
    model_kwargs, using_4bit = build_model_init_kwargs(
        config,
        compute_dtype,
        enable_4bit=enable_4bit,
    )
    tokenizer = load_tokenizer(model_source, config)
    model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer, {
        "model_source": model_source,
        "precision": precision_name,
        "using_4bit": using_4bit,
    }


def prepare_model_for_training(model, gradient_checkpointing: bool = True):
    is_quantized = bool(
        getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False)
    )
    if is_quantized:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=gradient_checkpointing,
        )
    elif gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    return model


def build_lora_config(config: dict[str, Any]) -> LoraConfig:
    lora_cfg = config["lora"]
    return LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
        target_modules=lora_cfg["target_modules"],
    )


def build_chat_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def batch_to_model_device(batch: dict[str, torch.Tensor], model) -> dict[str, torch.Tensor]:
    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device
    return {key: value.to(device) for key, value in batch.items()}


def trainable_parameter_summary(model) -> str:
    if hasattr(model, "get_nb_trainable_parameters"):
        trainable, total = model.get_nb_trainable_parameters()
        pct = 100.0 * trainable / total if total else 0.0
        return f"trainable_params={trainable:,} total_params={total:,} trainable_pct={pct:.4f}"
    trainable = 0
    total = 0
    for parameter in model.parameters():
        total += parameter.numel()
        if parameter.requires_grad:
            trainable += parameter.numel()
    pct = 100.0 * trainable / total if total else 0.0
    return f"trainable_params={trainable:,} total_params={total:,} trainable_pct={pct:.4f}"


def import_version(module_name: str) -> str:
    module = importlib.import_module(module_name)
    return getattr(module, "__version__", "unknown")


def query_nvidia_smi() -> dict[str, str]:
    if not torch.cuda.is_available():
        return {}
    fields = "name,memory.total,driver_version,compute_cap"
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    values = [item.strip() for item in result.stdout.strip().split(",")]
    keys = ["gpu_name", "gpu_memory_total", "driver_version", "compute_capability"]
    return dict(zip(keys, values))


def collect_environment_info(config: dict[str, Any]) -> dict[str, Any]:
    compute_dtype, precision_name = detect_compute_dtype(config)
    info: dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": import_version("torch"),
        "transformers_version": import_version("transformers"),
        "trl_version": import_version("trl"),
        "peft_version": import_version("peft"),
        "datasets_version": import_version("datasets"),
        "accelerate_version": import_version("accelerate"),
        "bitsandbytes_version": import_version("bitsandbytes"),
        "numpy_version": import_version("numpy"),
        "cuda_available": torch.cuda.is_available(),
        "compute_dtype": precision_name,
    }
    if torch.cuda.is_available():
        info["torch_cuda_version"] = torch.version.cuda
        info["bf16_supported"] = torch.cuda.is_bf16_supported()
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
        info.update(query_nvidia_smi())
    else:
        info["torch_cuda_version"] = None
        info["bf16_supported"] = False
        info["gpu_count"] = 0
    return info


def python_executable() -> str:
    return sys.executable
