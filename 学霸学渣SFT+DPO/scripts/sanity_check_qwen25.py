#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training_utils import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    batch_to_model_device,
    build_chat_prompt,
    collect_environment_info,
    load_config,
    load_model_and_tokenizer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal Qwen2.5-3B-Instruct smoke test.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the shared training config file.",
    )
    parser.add_argument(
        "--disable-4bit",
        action="store_true",
        help="Disable 4-bit loading for the smoke test.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path to write the smoke test result as JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    env_info = collect_environment_info(config)

    result: dict[str, object] = {
        "model_hub_id": config["model"]["hub_id"],
        "forward_ok": False,
        "generate_ok": False,
        "generated_text": "",
        "error": None,
        "environment": env_info,
    }

    model = None
    try:
        model, tokenizer, load_info = load_model_and_tokenizer(
            config,
            enable_4bit=not args.disable_4bit,
        )
        result.update(load_info)
        if not config["smoke_test"]["do_sample"]:
            model.generation_config.do_sample = False
            model.generation_config.temperature = None
            model.generation_config.top_p = None
            model.generation_config.top_k = None

        prompt_text = build_chat_prompt(tokenizer, config["smoke_test"]["messages"])
        batch = tokenizer(prompt_text, return_tensors="pt")
        batch = batch_to_model_device(batch, model)

        with torch.inference_mode():
            outputs = model(**batch)
        logits_shape = list(outputs.logits.shape)
        result["forward_ok"] = True
        result["forward_logits_shape"] = logits_shape

        with torch.inference_mode():
            generated = model.generate(
                **batch,
                max_new_tokens=config["smoke_test"]["max_new_tokens"],
                do_sample=config["smoke_test"]["do_sample"],
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_length = batch["input_ids"].shape[1]
        completion_ids = generated[0][prompt_length:]
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        result["generate_ok"] = True
        result["generated_text"] = completion_text
    except Exception as exc:  # pragma: no cover - used for runtime diagnostics
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["forward_ok"] and result["generate_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
