#!/usr/bin/env python

from __future__ import annotations

import argparse
import copy
import json
import random
import statistics
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training_utils import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    build_lora_config,
    ensure_dir,
    load_config,
    load_model_and_tokenizer,
    prepare_model_for_training,
    resolve_project_path,
    trainable_parameter_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SFT variants with Qwen2.5-3B-Instruct.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--variant",
        choices=("cot", "nocot"),
        default="cot",
        help="Which SFT dataset variant to train.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Root output directory for this SFT run. Defaults to outputs/sft_<variant>_qwen25_3b.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default="auto",
        help="Checkpoint path to resume from, or 'auto' to detect the latest checkpoint.",
    )
    parser.add_argument(
        "--disable-4bit",
        action="store_true",
        help="Disable 4-bit QLoRA loading.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=2.0,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="LoRA learning rate.",
    )
    parser.add_argument(
        "--sample-check-count",
        type=int,
        default=3,
        help="How many train samples to print for format checking before training.",
    )
    parser.add_argument(
        "--val-preview-count",
        type=int,
        default=5,
        help="How many validation samples to generate after training.",
    )
    return parser.parse_args()


def validate_messages(example: dict[str, Any]) -> None:
    messages = example.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError(f"Invalid messages field for id={example.get('id')}")
    expected_roles = ["user", "assistant"]
    roles = [message.get("role") for message in messages]
    if roles != expected_roles:
        raise ValueError(f"Unexpected roles for id={example.get('id')}: {roles}")
    for message in messages:
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError(f"Empty content for id={example.get('id')}")


def sample_format_checks(dataset_split, sample_count: int, seed: int) -> list[dict[str, Any]]:
    total = len(dataset_split)
    chosen = min(sample_count, total)
    indices = random.Random(seed).sample(range(total), k=chosen) if chosen else []
    samples = [dataset_split[index] for index in indices]
    for example in samples:
        validate_messages(example)
    return samples


def preview_text(text: str, limit: int = 180) -> str:
    compact = " ".join(text.strip().split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def assistant_content(example: dict[str, Any]) -> str:
    validate_messages(example)
    return example["messages"][1]["content"]


def dataset_length_summary(dataset_split, tokenizer) -> dict[str, Any]:
    char_lengths: list[int] = []
    word_lengths: list[int] = []
    token_lengths: list[int] = []

    for example in dataset_split:
        text = assistant_content(example)
        char_lengths.append(len(text))
        word_lengths.append(len(text.split()))
        token_lengths.append(len(tokenizer(text, add_special_tokens=False)["input_ids"]))

    def summary(values: list[int]) -> dict[str, float | int]:
        return {
            "avg": statistics.mean(values) if values else 0.0,
            "median": statistics.median(values) if values else 0.0,
            "min": min(values) if values else 0,
            "max": max(values) if values else 0,
        }

    return {
        "count": len(dataset_split),
        "chars": summary(char_lengths),
        "words": summary(word_lengths),
        "tokens": summary(token_lengths),
    }


def write_dataset_length_comparison(
    path: Path,
    *,
    cot_stats: dict[str, Any],
    nocot_stats: dict[str, Any],
) -> None:
    avg_token_delta = nocot_stats["tokens"]["avg"] - cot_stats["tokens"]["avg"]
    avg_char_delta = nocot_stats["chars"]["avg"] - cot_stats["chars"]["avg"]
    lines = [
        "# SFT Dataset Length Comparison",
        "",
        "- Source splits: `sft_cot_train.jsonl` vs `sft_nocot_train.jsonl`",
        "- Lengths are measured on assistant message content only, without chat template wrapper.",
        "",
        "| Dataset | Count | Avg Chars | Avg Words | Avg Tokens | Median Tokens | Min Tokens | Max Tokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| SFT-CoT Train | {cot_stats['count']} | "
            f"{cot_stats['chars']['avg']:.2f} | {cot_stats['words']['avg']:.2f} | "
            f"{cot_stats['tokens']['avg']:.2f} | {cot_stats['tokens']['median']:.2f} | "
            f"{cot_stats['tokens']['min']} | {cot_stats['tokens']['max']} |"
        ),
        (
            f"| SFT-NoCoT Train | {nocot_stats['count']} | "
            f"{nocot_stats['chars']['avg']:.2f} | {nocot_stats['words']['avg']:.2f} | "
            f"{nocot_stats['tokens']['avg']:.2f} | {nocot_stats['tokens']['median']:.2f} | "
            f"{nocot_stats['tokens']['min']} | {nocot_stats['tokens']['max']} |"
        ),
        "",
        "## Delta",
        "",
        f"- Avg assistant chars delta (`NoCoT - CoT`): `{avg_char_delta:.2f}`",
        f"- Avg assistant tokens delta (`NoCoT - CoT`): `{avg_token_delta:.2f}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_resume_checkpoint(output_dir: Path, resume_arg: str | None) -> str | None:
    if not resume_arg:
        return None
    if resume_arg == "auto":
        if output_dir.exists():
            return get_last_checkpoint(str(output_dir))
        return None
    return str(Path(resume_arg).resolve())


def extract_answer_from_text(text: str) -> str:
    lower = text.lower()
    start = lower.rfind("<answer>")
    end = lower.rfind("</answer>")
    if start != -1 and end != -1 and start < end:
        return text[start + len("<answer>"):end].strip()
    return text.strip()


def run_val_preview(
    trainer: SFTTrainer,
    tokenizer,
    eval_dataset,
    count: int,
    output_path: Path,
    seed: int,
) -> list[dict[str, Any]]:
    indices = random.Random(seed).sample(range(len(eval_dataset)), k=min(count, len(eval_dataset)))
    previews: list[dict[str, Any]] = []

    trainer.model.eval()
    trainer.model.generation_config.do_sample = False
    trainer.model.generation_config.temperature = None
    trainer.model.generation_config.top_p = None
    trainer.model.generation_config.top_k = None

    with output_path.open("w", encoding="utf-8") as handle:
        for index in indices:
            example = eval_dataset[index]
            prompt_text = tokenizer.apply_chat_template(
                [example["messages"][0]],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(prompt_text, return_tensors="pt")
            device = next(trainer.model.parameters()).device
            inputs = {key: value.to(device) for key, value in inputs.items()}
            autocast_dtype = None
            if device.type == "cuda":
                if getattr(trainer.args, "bf16", False):
                    autocast_dtype = torch.bfloat16
                elif getattr(trainer.args, "fp16", False):
                    autocast_dtype = torch.float16
            autocast_context = (
                torch.autocast(device_type=device.type, dtype=autocast_dtype)
                if autocast_dtype is not None
                else nullcontext()
            )
            with torch.inference_mode(), autocast_context:
                generated = trainer.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            prompt_tokens = inputs["input_ids"].shape[1]
            completion_ids = generated[0][prompt_tokens:]
            generated_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
            record = {
                "id": example["id"],
                "question": example["messages"][0]["content"],
                "gold_answer": example.get("gold_answer"),
                "reference_assistant": example["messages"][1]["content"],
                "generated_text": generated_text,
                "extracted_final_answer": extract_answer_from_text(generated_text),
            }
            previews.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return previews


def write_markdown_log(
    path: Path,
    *,
    run_label: str,
    started_at: datetime,
    finished_at: datetime,
    output_dir: Path,
    train_count: int,
    val_count: int,
    sample_checks: list[dict[str, Any]],
    load_info: dict[str, Any],
    trainable_summary: str,
    resume_checkpoint: str | None,
    trainer: SFTTrainer,
    final_metrics: dict[str, Any],
    val_previews: list[dict[str, Any]],
) -> None:
    state = trainer.state
    lines = [
        f"# {run_label} Train Log",
        "",
        f"- Started: `{started_at.isoformat()}`",
        f"- Finished: `{finished_at.isoformat()}`",
        f"- Output dir: `{output_dir}`",
        f"- Model source: `{load_info['model_source']}`",
        f"- Precision: `{load_info['precision']}`",
        f"- Using 4-bit: `{load_info['using_4bit']}`",
        f"- Train samples: `{train_count}`",
        f"- Val samples: `{val_count}`",
        f"- Resume checkpoint: `{resume_checkpoint}`",
        f"- Best checkpoint: `{state.best_model_checkpoint}`",
        f"- Best metric: `{state.best_metric}`",
        f"- Global step: `{state.global_step}`",
        f"- Trainable params: `{trainable_summary}`",
        "",
        "## Sample Checks",
        "",
    ]

    for example in sample_checks:
        lines.extend(
            [
                f"- id: `{example['id']}`",
                f"  roles: `{[message['role'] for message in example['messages']]}`",
                f"  user: `{preview_text(example['messages'][0]['content'])}`",
                f"  assistant: `{preview_text(example['messages'][1]['content'])}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Final Metrics",
            "",
        ]
    )
    for key, value in sorted(final_metrics.items()):
        lines.append(f"- {key}: `{value}`")

    lines.extend(
        [
            "",
            "## Validation Preview",
            "",
        ]
    )
    for record in val_previews:
        lines.extend(
            [
                f"- id: `{record['id']}`",
                f"  gold: `{preview_text(str(record['gold_answer']), 120)}`",
                f"  pred: `{preview_text(record['extracted_final_answer'], 120)}`",
            ]
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    runtime_cfg = config["runtime"]
    paths_cfg = config["paths"]
    sft_cfg = copy.deepcopy(config["sft"])
    variant = args.variant
    run_label = f"SFT-{variant.upper()}"

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (PROJECT_ROOT / "outputs" / f"sft_{variant}_qwen25_3b").resolve()
    )
    logging_dir = ensure_dir(output_dir / "logs")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = resolve_project_path(paths_cfg[f"sft_{variant}_train"])
    eval_path = resolve_project_path(paths_cfg[f"sft_{variant}_val"])
    dataset = load_dataset(
        "json",
        data_files={"train": str(train_path), "validation": str(eval_path)},
    )
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    sample_checks = sample_format_checks(
        train_dataset,
        sample_count=args.sample_check_count,
        seed=runtime_cfg["seed"],
    )

    print(f"train_samples={len(train_dataset)}")
    print(f"val_samples={len(eval_dataset)}")
    print(f"variant={variant}")
    print("sample_format_checks:")
    for example in sample_checks:
        print(f"  id={example['id']} roles={[message['role'] for message in example['messages']]}")
        print(f"  user={preview_text(example['messages'][0]['content'])}")
        print(f"  assistant={preview_text(example['messages'][1]['content'])}")

    model, tokenizer, load_info = load_model_and_tokenizer(
        config,
        enable_4bit=not args.disable_4bit,
    )
    model = prepare_model_for_training(model, gradient_checkpointing=True)
    tokenizer.save_pretrained(output_dir)

    if variant == "nocot":
        cot_train_path = resolve_project_path(paths_cfg["sft_cot_train"])
        cot_train_dataset = load_dataset(
            "json",
            data_files={"train": str(cot_train_path)},
        )["train"]
        cot_length_stats = dataset_length_summary(cot_train_dataset, tokenizer)
        nocot_length_stats = dataset_length_summary(train_dataset, tokenizer)
        comparison_path = output_dir / "sft_dataset_length_comparison.md"
        write_dataset_length_comparison(
            comparison_path,
            cot_stats=cot_length_stats,
            nocot_stats=nocot_length_stats,
        )
        print(f"dataset_length_comparison_path={comparison_path}")
        print(
            "dataset_length_tokens_avg="
            f"cot:{cot_length_stats['tokens']['avg']:.2f} "
            f"nocot:{nocot_length_stats['tokens']['avg']:.2f}"
        )

    trainer_args = SFTConfig(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        logging_dir=str(logging_dir),
        report_to="none",
        seed=runtime_cfg["seed"],
        data_seed=runtime_cfg["seed"],
        bf16=load_info["precision"] == "bf16",
        fp16=load_info["precision"] == "fp16",
        per_device_train_batch_size=sft_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=sft_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=sft_cfg["gradient_accumulation_steps"],
        learning_rate=float(args.learning_rate),
        num_train_epochs=float(args.num_train_epochs),
        warmup_ratio=float(sft_cfg["warmup_ratio"]),
        logging_steps=5,
        save_steps=20,
        eval_steps=20,
        save_strategy="steps",
        eval_strategy="steps",
        save_total_limit=2,
        save_safetensors=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_length=int(sft_cfg["max_length"]),
        packing=bool(sft_cfg["packing"]),
        assistant_only_loss=bool(sft_cfg["assistant_only_loss"]),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        weight_decay=0.0,
        max_grad_norm=1.0,
    )

    training_args_path = output_dir / f"training_args_sft_{variant}.json"
    training_args_path.write_text(trainer_args.to_json_string(), encoding="utf-8")
    save_json(
        output_dir / f"resolved_run_config_sft_{variant}.json",
        {
            "base_config": config,
            "cli_args": vars(args),
            "effective_training_args": json.loads(trainer_args.to_json_string()),
        },
    )

    trainer = SFTTrainer(
        model=model,
        args=trainer_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=build_lora_config(config),
    )

    resume_checkpoint = resolve_resume_checkpoint(output_dir, args.resume_from_checkpoint)
    trainable_summary = trainable_parameter_summary(trainer.model)
    print(f"model_source={load_info['model_source']}")
    print(f"precision={load_info['precision']}")
    print(f"using_4bit={load_info['using_4bit']}")
    print(f"run_label={run_label}")
    print(f"output_dir={output_dir}")
    print(f"resume_checkpoint={resume_checkpoint}")
    print(trainable_summary)

    started_at = datetime.now()
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    final_metrics = dict(train_result.metrics)
    eval_metrics = trainer.evaluate()
    final_metrics.update(eval_metrics)

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)
    trainer.save_state()

    val_preview_path = output_dir / f"sft_{variant}_val_preview.jsonl"
    val_previews = run_val_preview(
        trainer=trainer,
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,
        count=args.val_preview_count,
        output_path=val_preview_path,
        seed=runtime_cfg["seed"] + 1,
    )
    finished_at = datetime.now()

    write_markdown_log(
        path=output_dir / f"sft_{variant}_train_log.md",
        run_label=run_label,
        started_at=started_at,
        finished_at=finished_at,
        output_dir=output_dir,
        train_count=len(train_dataset),
        val_count=len(eval_dataset),
        sample_checks=sample_checks,
        load_info=load_info,
        trainable_summary=trainable_summary,
        resume_checkpoint=resume_checkpoint,
        trainer=trainer,
        final_metrics=final_metrics,
        val_previews=val_previews,
    )

    print("training_complete=true")
    print(f"best_checkpoint={trainer.state.best_model_checkpoint}")
    print(f"trainer_state_path={output_dir / 'trainer_state.json'}")
    print(f"training_args_path={training_args_path}")
    print(f"val_preview_path={val_preview_path}")
    print(json.dumps(final_metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
