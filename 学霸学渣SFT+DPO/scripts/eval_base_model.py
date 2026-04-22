#!/usr/bin/env python

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import unicodedata
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training_utils import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    batch_to_model_device,
    load_config,
    load_model_and_tokenizer,
    resolve_project_path,
)

DIRECT_SYSTEM_PROMPT = (
    "You are solving evaluation questions. Follow the user's formatting constraints exactly. "
    "Give only a concise final answer wrapped in <answer> and </answer>. Do not explain."
)

COT_SYSTEM_PROMPT = (
    "You are solving evaluation questions. Follow the user's formatting constraints exactly. "
    "Reason briefly in 1 to 4 short steps inside <think>, then put the final answer in "
    "<answer> and </answer>."
)

ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
FINAL_ANSWER_RE = re.compile(
    r"(?:final answer|answer)\s*[:：]\s*(.+)",
    re.IGNORECASE | re.DOTALL,
)
THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]+)\}")
FRACTION_RE = re.compile(r"\\(?:d?frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluation for base or LoRA-adapted Qwen2.5 models.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--mode",
        choices=("direct", "cot", "default", "both"),
        default="both",
        help="Which evaluation mode to run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "baselines",
        help="Directory for prediction and report outputs.",
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=None,
        help="Optional LoRA adapter or checkpoint path to load on top of the base model.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="base",
        help="Prefix for prediction, stats, and report filenames.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional sample limit for debugging.",
    )
    parser.add_argument(
        "--disable-4bit",
        action="store_true",
        help="Disable 4-bit QLoRA-style loading for evaluation.",
    )
    parser.add_argument(
        "--direct-max-new-tokens",
        type=int,
        default=128,
        help="Generation cap for direct mode.",
    )
    parser.add_argument(
        "--cot-max-new-tokens",
        type=int,
        default=160,
        help="Generation cap for cot mode.",
    )
    return parser.parse_args()


def load_test_examples(config: dict[str, Any], limit: int | None = None) -> list[dict[str, Any]]:
    test_path = resolve_project_path(config["paths"]["test_eval"])
    examples: list[dict[str, Any]] = []
    with test_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                examples.append(json.loads(line))
            if limit is not None and len(examples) >= limit:
                break
    return examples


def make_messages(mode: str, question: str) -> list[dict[str, str]]:
    if mode == "default":
        return [{"role": "user", "content": question}]
    system_prompt = DIRECT_SYSTEM_PROMPT if mode == "direct" else COT_SYSTEM_PROMPT
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def extract_final_answer(raw_output: str) -> str:
    text = raw_output.strip()
    matches = ANSWER_TAG_RE.findall(text)
    if matches:
        return matches[-1].strip()

    final_matches = FINAL_ANSWER_RE.findall(text)
    if final_matches:
        return final_matches[-1].strip()

    text = THINK_TAG_RE.sub("", text).strip()
    text = re.sub(r"```(?:[\w+-]+)?", "", text).strip()
    text = re.sub(r"</?[^>]+>", "", text).strip()
    non_empty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if non_empty_lines:
        candidate = non_empty_lines[-1]
    else:
        candidate = text
    candidate = re.sub(r"^(?:final answer|answer)\s*[:：]\s*", "", candidate, flags=re.IGNORECASE)
    return candidate.strip()


def latex_fraction_to_plain(text: str) -> str:
    previous = None
    current = text
    while previous != current:
        previous = current
        current = FRACTION_RE.sub(r"\1/\2", current)
    return current


def normalize_answer(text: str) -> str:
    text = html.unescape(text or "")
    text = unicodedata.normalize("NFKC", text)
    text = text.strip()
    text = BOXED_RE.sub(r"\1", text)
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("$", "")
    text = latex_fraction_to_plain(text)
    text = text.replace("→", "->")
    text = re.sub(r"\s*->\s*", "->", text)
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = text.replace("：", ":")
    text = re.sub(r"\s*,\s*", ",", text)
    text = re.sub(r"\s*°", "°", text)
    text = re.sub(r"\s*/\s*", "/", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip().strip('"\'.')
    return text.casefold()


def answers_match(prediction: str, gold_answer: str, question_type: str) -> bool:
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(gold_answer)
    if not pred_norm:
        return False
    if pred_norm == gold_norm:
        return True
    if question_type == "open_ended":
        shorter, longer = sorted((pred_norm, gold_norm), key=len)
        if len(shorter) >= 6 and shorter in longer:
            return True
    return False


def output_path_for_mode(output_dir: Path, prefix: str, mode: str) -> Path:
    return output_dir / f"{prefix}_{mode}_predictions.jsonl"


def generate_one(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    max_new_tokens: int,
    precision: str,
) -> tuple[str, str, int]:
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = batch_to_model_device(inputs, model)
    device = inputs["input_ids"].device
    autocast_dtype = None
    if device.type == "cuda":
        if precision == "bf16":
            autocast_dtype = torch.bfloat16
        elif precision == "fp16":
            autocast_dtype = torch.float16
    autocast_context = (
        torch.autocast(device_type=device.type, dtype=autocast_dtype)
        if autocast_dtype is not None
        else nullcontext()
    )
    with torch.inference_mode(), autocast_context:
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_tokens = inputs["input_ids"].shape[1]
    completion_ids = generated[0][prompt_tokens:]
    raw_output = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    return prompt_text, raw_output, int(completion_ids.shape[0])


def evaluate_mode(
    model,
    tokenizer,
    examples: list[dict[str, Any]],
    mode: str,
    max_new_tokens: int,
    output_path: Path,
    precision: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for example in tqdm(examples, desc=f"Evaluating {mode}", ncols=100):
            messages = make_messages(mode, example["question"])
            prompt_text, raw_output, output_token_count = generate_one(
                model=model,
                tokenizer=tokenizer,
                messages=messages,
                max_new_tokens=max_new_tokens,
                precision=precision,
            )
            extracted = extract_final_answer(raw_output)
            is_correct = answers_match(extracted, example["gold_answer"], example["question_type"])

            record = {
                "id": example["id"],
                "prompt": prompt_text,
                "raw_output": raw_output,
                "extracted_final_answer": extracted,
                "gold_answer": example["gold_answer"],
                "difficulty": example["difficulty"],
                "subset": example["subset"],
                "question_type": example["question_type"],
                "mode": mode,
                "output_token_count": output_token_count,
                "is_correct": is_correct,
            }
            predictions.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

    stats = compute_mode_stats(predictions)
    return predictions, stats


def subset_stats(predictions: list[dict[str, Any]], difficulty: str | None = None) -> dict[str, Any]:
    filtered = [
        record
        for record in predictions
        if difficulty is None or record["difficulty"] == difficulty
    ]
    total = len(filtered)
    correct = sum(1 for record in filtered if record["is_correct"])
    avg_tokens = (
        sum(record["output_token_count"] for record in filtered) / total
        if total
        else 0.0
    )
    accuracy = correct / total if total else 0.0
    return {
        "count": total,
        "correct": correct,
        "accuracy": accuracy,
        "avg_output_tokens": avg_tokens,
    }


def compute_mode_stats(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    simple = subset_stats(predictions, difficulty="simple")
    complex_stats = subset_stats(predictions, difficulty="complex")
    overall = subset_stats(predictions, difficulty=None)
    return {
        "simple_count": simple["count"],
        "complex_count": complex_stats["count"],
        "overall_count": overall["count"],
        "simple_accuracy": simple["accuracy"],
        "complex_accuracy": complex_stats["accuracy"],
        "overall_accuracy": overall["accuracy"],
        "simple_avg_output_tokens": simple["avg_output_tokens"],
        "complex_avg_output_tokens": complex_stats["avg_output_tokens"],
        "overall_avg_output_tokens": overall["avg_output_tokens"],
        "simple_correct": simple["correct"],
        "complex_correct": complex_stats["correct"],
        "overall_correct": overall["correct"],
    }


def render_report(
    stats_by_mode: dict[str, dict[str, Any]],
    output_dir: Path,
    config: dict[str, Any],
    model_source: str,
    output_prefix: str,
    adapter_path: str | None,
) -> str:
    lines = [
        "# Evaluation Report",
        "",
        f"- Base model: `{config['model']['hub_id']}`",
        f"- Model source used: `{model_source}`",
        f"- Adapter path: `{adapter_path}`",
        f"- Test set: `{config['paths']['test_eval']}`",
        f"- Output directory: `{output_dir}`",
        "",
        "## Metric",
        "",
        "- Accuracy is computed from `extracted_final_answer` against `gold_answer` after normalization.",
        "- Open-ended items additionally allow normalized containment matching.",
        "",
        "## Summary",
        "",
        "| Mode | Simple Accuracy | Complex Accuracy | Overall Accuracy | Simple Avg Tokens | Complex Avg Tokens | Overall Avg Tokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for mode in ("direct", "cot"):
        if mode not in stats_by_mode:
            continue
        stats = stats_by_mode[mode]
        lines.append(
            "| "
            f"{mode} | "
            f"{stats['simple_accuracy']:.4f} | "
            f"{stats['complex_accuracy']:.4f} | "
            f"{stats['overall_accuracy']:.4f} | "
            f"{stats['simple_avg_output_tokens']:.2f} | "
            f"{stats['complex_avg_output_tokens']:.2f} | "
            f"{stats['overall_avg_output_tokens']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Direct predictions: `{output_path_for_mode(output_dir, output_prefix, 'direct').name}`",
            f"- CoT predictions: `{output_path_for_mode(output_dir, output_prefix, 'cot').name}`",
            f"- Stats JSON: `{output_prefix}_eval_stats.json`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    examples = load_test_examples(config, limit=args.limit)
    model, tokenizer, load_info = load_model_and_tokenizer(
        config,
        enable_4bit=not args.disable_4bit,
    )
    adapter_path = None
    if args.adapter_path is not None:
        adapter_path = str(args.adapter_path.resolve())
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    modes = ["direct", "cot"] if args.mode == "both" else [args.mode]
    max_new_tokens_by_mode = {
        "direct": args.direct_max_new_tokens,
        "cot": args.cot_max_new_tokens,
    }
    stats_by_mode: dict[str, dict[str, Any]] = {}
    for mode in modes:
        _, stats = evaluate_mode(
            model=model,
            tokenizer=tokenizer,
            examples=examples,
            mode=mode,
            max_new_tokens=max_new_tokens_by_mode[mode],
            output_path=output_path_for_mode(output_dir, args.output_prefix, mode),
            precision=load_info["precision"],
        )
        stats_by_mode[mode] = stats

    stats_payload = {
        "output_prefix": args.output_prefix,
        "model_hub_id": config["model"]["hub_id"],
        "model_source": load_info["model_source"],
        "adapter_path": adapter_path,
        "precision": load_info["precision"],
        "using_4bit": load_info["using_4bit"],
        "test_count": len(examples),
        "modes": stats_by_mode,
    }
    stats_path = output_dir / f"{args.output_prefix}_eval_stats.json"
    stats_path.write_text(
        json.dumps(stats_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_path = output_dir / f"{args.output_prefix}_eval_report.md"
    report_path.write_text(
        render_report(
            stats_by_mode,
            output_dir,
            config,
            load_info["model_source"],
            args.output_prefix,
            adapter_path,
        ),
        encoding="utf-8",
    )

    print(json.dumps(stats_payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
