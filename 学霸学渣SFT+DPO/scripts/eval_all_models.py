#!/usr/bin/env python

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval_base_model import (  # noqa: E402
    compute_mode_stats,
    evaluate_mode,
    load_test_examples,
)
from src.training_utils import load_config, load_model_and_tokenizer  # noqa: E402


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "final_eval"
DEFAULT_BASELINE_STATS = PROJECT_ROOT / "outputs" / "baselines" / "base_eval_stats.json"
DEFAULT_SFT_COT = PROJECT_ROOT / "outputs" / "sft_cot_qwen25_3b"
DEFAULT_SFT_NOCOT = PROJECT_ROOT / "outputs" / "sft_nocot_qwen25_3b"
DEFAULT_DPO = PROJECT_ROOT / "outputs" / "dpo_adaptive_qwen25_3b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified evaluation for base, SFT, and DPO models.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "base.yaml")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for final evaluation tables and prediction files.",
    )
    parser.add_argument(
        "--baseline-stats-path",
        type=Path,
        default=DEFAULT_BASELINE_STATS,
        help="Optional existing base baseline stats JSON to reuse.",
    )
    parser.add_argument(
        "--disable-4bit",
        action="store_true",
        help="Disable 4-bit loading during evaluation.",
    )
    parser.add_argument(
        "--default-max-new-tokens",
        type=int,
        default=256,
        help="Generation cap for user-only/default-prompt evaluation rows.",
    )
    parser.add_argument(
        "--sft-cot-adapter-path",
        type=Path,
        default=DEFAULT_SFT_COT,
        help="Path to the exported SFT-CoT adapter directory.",
    )
    parser.add_argument(
        "--sft-nocot-adapter-path",
        type=Path,
        default=DEFAULT_SFT_NOCOT,
        help="Path to the exported SFT-NoCoT adapter directory.",
    )
    parser.add_argument(
        "--dpo-adapter-path",
        type=Path,
        default=DEFAULT_DPO,
        help="Path to the exported DPO adapter directory.",
    )
    return parser.parse_args()


def resolve_best_export(path: Path) -> Path:
    if path.is_file():
        return path
    trainer_state = path / "trainer_state.json"
    if trainer_state.exists():
        payload = json.loads(trainer_state.read_text(encoding="utf-8"))
        best_checkpoint = payload.get("best_model_checkpoint")
        if best_checkpoint:
            return Path(best_checkpoint)
    return path


def evaluate_spec(
    *,
    config: dict[str, Any],
    output_dir: Path,
    output_prefix: str,
    mode: str,
    label: str,
    adapter_path: Path | None,
    examples: list[dict[str, Any]],
    max_new_tokens: int,
    disable_4bit: bool,
) -> dict[str, Any]:
    model, tokenizer, load_info = load_model_and_tokenizer(config, enable_4bit=not disable_4bit)
    resolved_adapter_path = None
    if adapter_path is not None:
        resolved_adapter_path = str(resolve_best_export(adapter_path).resolve())
        model = PeftModel.from_pretrained(model, resolved_adapter_path)
    model.eval()
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    predictions_path = output_dir / f"{output_prefix}_predictions.jsonl"
    _, stats = evaluate_mode(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        mode=mode,
        max_new_tokens=max_new_tokens,
        output_path=predictions_path,
        precision=load_info["precision"],
    )
    stats["length_gap"] = stats["complex_avg_output_tokens"] - stats["simple_avg_output_tokens"]
    stats.update(
        {
            "model_label": label,
            "model_key": output_prefix,
            "prompt_mode": mode,
            "adapter_path": resolved_adapter_path,
            "model_source": load_info["model_source"],
            "precision": load_info["precision"],
            "using_4bit": load_info["using_4bit"],
            "predictions_path": str(predictions_path),
        }
    )

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return stats


def load_reused_base_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: dict[str, dict[str, Any]] = {}
    for key, label in [("direct", "Base direct"), ("cot", "Base cot")]:
        if key not in payload.get("modes", {}):
            continue
        stats = dict(payload["modes"][key])
        stats["length_gap"] = stats["complex_avg_output_tokens"] - stats["simple_avg_output_tokens"]
        stats.update(
            {
                "model_label": label,
                "model_key": f"base_{key}",
                "prompt_mode": key,
                "adapter_path": None,
                "model_source": payload.get("model_source"),
                "precision": payload.get("precision"),
                "using_4bit": payload.get("using_4bit"),
                "predictions_path": None,
                "reused_from": str(path),
            }
        )
        rows[key] = stats
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "model_label",
        "model_key",
        "prompt_mode",
        "simple_accuracy",
        "complex_accuracy",
        "overall_accuracy",
        "simple_avg_output_tokens",
        "complex_avg_output_tokens",
        "overall_avg_output_tokens",
        "length_gap",
        "simple_correct",
        "complex_correct",
        "overall_correct",
        "simple_count",
        "complex_count",
        "overall_count",
        "adapter_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def choose_most_adaptive(rows: list[dict[str, Any]]) -> dict[str, Any]:
    neutral_rows = [row for row in rows if row["prompt_mode"] == "default"]
    if not neutral_rows:
        return max(rows, key=lambda row: row["length_gap"])
    return max(neutral_rows, key=lambda row: (row["length_gap"], row["overall_accuracy"]))


def write_report(path: Path, rows: list[dict[str, Any]], most_adaptive: dict[str, Any]) -> None:
    lines = [
        "# Final Results Report",
        "",
        "## Prompt Modes",
        "",
        "- `Base direct`: explicit concise-answer prompt.",
        "- `Base cot`: explicit think-then-answer prompt.",
        "- `SFT-CoT`, `SFT-NoCoT`, `DPO-Adaptive`: default user-only chat prompt so the model chooses its own response length.",
        "",
        "## Summary Table",
        "",
        "| Model | Prompt Mode | Simple Accuracy | Complex Accuracy | Overall Accuracy | Simple Avg Tokens | Complex Avg Tokens | Overall Avg Tokens | Length Gap |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in rows:
        lines.append(
            "| "
            f"{row['model_label']} | "
            f"{row['prompt_mode']} | "
            f"{row['simple_accuracy']:.4f} | "
            f"{row['complex_accuracy']:.4f} | "
            f"{row['overall_accuracy']:.4f} | "
            f"{row['simple_avg_output_tokens']:.2f} | "
            f"{row['complex_avg_output_tokens']:.2f} | "
            f"{row['overall_avg_output_tokens']:.2f} | "
            f"{row['length_gap']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Most adaptive by length gap under the default prompt: `{most_adaptive['model_label']}`.",
            (
                f"- It shows a length gap of `{most_adaptive['length_gap']:.2f}` tokens "
                f"with overall accuracy `{most_adaptive['overall_accuracy']:.4f}`."
            ),
            "- A larger positive length gap means the model spends more tokens on complex items than on simple items.",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    examples = load_test_examples(config)

    rows: list[dict[str, Any]] = []
    reused_base_rows = load_reused_base_rows(args.baseline_stats_path.resolve())
    if reused_base_rows:
        rows.append(reused_base_rows["direct"])
        rows.append(reused_base_rows["cot"])
    else:
        rows.append(
            evaluate_spec(
                config=config,
                output_dir=output_dir,
                output_prefix="base_direct",
                mode="direct",
                label="Base direct",
                adapter_path=None,
                examples=examples,
                max_new_tokens=128,
                disable_4bit=args.disable_4bit,
            )
        )
        rows.append(
            evaluate_spec(
                config=config,
                output_dir=output_dir,
                output_prefix="base_cot",
                mode="cot",
                label="Base cot",
                adapter_path=None,
                examples=examples,
                max_new_tokens=160,
                disable_4bit=args.disable_4bit,
            )
        )

    rows.append(
        evaluate_spec(
            config=config,
            output_dir=output_dir,
            output_prefix="sft_cot",
            mode="default",
            label="SFT-CoT",
            adapter_path=args.sft_cot_adapter_path.resolve(),
            examples=examples,
            max_new_tokens=args.default_max_new_tokens,
            disable_4bit=args.disable_4bit,
        )
    )
    rows.append(
        evaluate_spec(
            config=config,
            output_dir=output_dir,
            output_prefix="sft_nocot",
            mode="default",
            label="SFT-NoCoT",
            adapter_path=args.sft_nocot_adapter_path.resolve(),
            examples=examples,
            max_new_tokens=args.default_max_new_tokens,
            disable_4bit=args.disable_4bit,
        )
    )
    rows.append(
        evaluate_spec(
            config=config,
            output_dir=output_dir,
            output_prefix="dpo_adaptive",
            mode="default",
            label="DPO-Adaptive",
            adapter_path=args.dpo_adapter_path.resolve(),
            examples=examples,
            max_new_tokens=args.default_max_new_tokens,
            disable_4bit=args.disable_4bit,
        )
    )

    final_results = {
        "rows": rows,
        "most_adaptive_model": choose_most_adaptive(rows),
    }

    json_path = output_dir / "final_results.json"
    json_path.write_text(json.dumps(final_results, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(output_dir / "final_results_table.csv", rows)
    write_report(output_dir / "final_results_report.md", rows, final_results["most_adaptive_model"])

    print(json.dumps(final_results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
