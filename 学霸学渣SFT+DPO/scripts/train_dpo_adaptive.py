#!/usr/bin/env python

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from peft import PeftModel
from transformers.trainer_utils import get_last_checkpoint
from trl import DPOConfig, DPOTrainer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval_base_model import extract_final_answer  # noqa: E402
from src.training_utils import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    load_config,
    load_model_and_tokenizer,
    prepare_model_for_training,
    resolve_project_path,
    trainable_parameter_summary,
)
from train_sft_cot import preview_text, save_json  # noqa: E402


DEFAULT_INIT_ADAPTER = PROJECT_ROOT / "outputs" / "sft_cot_qwen25_3b" / "checkpoint-80"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DPO-Adaptive from the SFT-CoT checkpoint.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "dpo_adaptive_qwen25_3b",
        help="Root output directory for the DPO run.",
    )
    parser.add_argument(
        "--init-adapter-path",
        type=Path,
        default=DEFAULT_INIT_ADAPTER,
        help="SFT-CoT adapter checkpoint used to initialize policy and reference models.",
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
        default=None,
        help="Override DPO epochs. Defaults to config value.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override DPO learning rate. Defaults to config value.",
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


def validate_dpo_example(example: dict[str, Any]) -> None:
    required_fields = ["id", "prompt", "chosen", "rejected", "difficulty", "subset", "question_type", "gold_answer"]
    for field in required_fields:
        if field not in example:
            raise ValueError(f"Missing field `{field}` in DPO example id={example.get('id')}")
    for field in ["prompt", "chosen", "rejected", "difficulty", "subset", "question_type", "gold_answer"]:
        value = example.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Invalid `{field}` in DPO example id={example.get('id')}")


def sample_format_checks(dataset_split, sample_count: int, seed: int) -> list[dict[str, Any]]:
    total = len(dataset_split)
    chosen = min(sample_count, total)
    indices = random.Random(seed).sample(range(total), k=chosen) if chosen else []
    samples = [dataset_split[index] for index in indices]
    for example in samples:
        validate_dpo_example(example)
    return samples


def chat_prompt_from_text(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def format_dpo_split(dataset_split: Dataset, tokenizer) -> Dataset:
    def _format_prompt(example: dict[str, Any]) -> dict[str, Any]:
        validate_dpo_example(example)
        return {"prompt": chat_prompt_from_text(tokenizer, example["prompt"])}

    return dataset_split.map(_format_prompt, desc="Formatting DPO prompts")


def resolve_resume_checkpoint(output_dir: Path, resume_arg: str | None) -> str | None:
    if not resume_arg:
        return None
    if resume_arg == "auto":
        if output_dir.exists():
            return get_last_checkpoint(str(output_dir))
        return None
    return str(Path(resume_arg).resolve())


def run_val_preview(
    trainer: DPOTrainer,
    tokenizer,
    raw_eval_examples: list[dict[str, Any]],
    count: int,
    output_path: Path,
    seed: int,
) -> list[dict[str, Any]]:
    indices = random.Random(seed).sample(range(len(raw_eval_examples)), k=min(count, len(raw_eval_examples)))
    previews: list[dict[str, Any]] = []

    trainer.model.eval()
    trainer.model.generation_config.do_sample = False
    trainer.model.generation_config.temperature = None
    trainer.model.generation_config.top_p = None
    trainer.model.generation_config.top_k = None

    with output_path.open("w", encoding="utf-8") as handle:
        for index in indices:
            example = raw_eval_examples[index]
            prompt_text = chat_prompt_from_text(tokenizer, example["prompt"])
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
                "difficulty": example["difficulty"],
                "subset": example["subset"],
                "question_type": example["question_type"],
                "prompt": example["prompt"],
                "chosen": example["chosen"],
                "rejected": example["rejected"],
                "gold_answer": example["gold_answer"],
                "generated_text": generated_text,
                "extracted_final_answer": extract_final_answer(generated_text),
            }
            previews.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return previews


def write_markdown_log(
    path: Path,
    *,
    started_at: datetime,
    finished_at: datetime,
    output_dir: Path,
    train_count: int,
    val_count: int,
    sample_checks: list[dict[str, Any]],
    load_info: dict[str, Any],
    init_adapter_path: Path,
    policy_summary: str,
    reference_summary: str,
    resume_checkpoint: str | None,
    trainer: DPOTrainer,
    final_metrics: dict[str, Any],
    val_previews: list[dict[str, Any]],
) -> None:
    state = trainer.state
    lines = [
        "# DPO-Adaptive Train Log",
        "",
        f"- Started: `{started_at.isoformat()}`",
        f"- Finished: `{finished_at.isoformat()}`",
        f"- Output dir: `{output_dir}`",
        f"- Model source: `{load_info['model_source']}`",
        f"- Precision: `{load_info['precision']}`",
        f"- Using 4-bit: `{load_info['using_4bit']}`",
        f"- Init adapter path: `{init_adapter_path}`",
        f"- Train samples: `{train_count}`",
        f"- Val samples: `{val_count}`",
        f"- Resume checkpoint: `{resume_checkpoint}`",
        f"- Best checkpoint: `{state.best_model_checkpoint}`",
        f"- Best metric: `{state.best_metric}`",
        f"- Global step: `{state.global_step}`",
        f"- Policy params: `{policy_summary}`",
        f"- Reference params: `{reference_summary}`",
        "",
        "## Sample Checks",
        "",
    ]

    for example in sample_checks:
        lines.extend(
            [
                f"- id: `{example['id']}`",
                f"  difficulty: `{example['difficulty']}` subset: `{example['subset']}` type: `{example['question_type']}`",
                f"  prompt: `{preview_text(example['prompt'])}`",
                f"  chosen: `{preview_text(example['chosen'])}`",
                f"  rejected: `{preview_text(example['rejected'])}`",
            ]
        )

    lines.extend(["", "## Final Metrics", ""])
    for key, value in sorted(final_metrics.items()):
        lines.append(f"- {key}: `{value}`")

    lines.extend(["", "## Validation Preview", ""])
    for record in val_previews:
        lines.extend(
            [
                f"- id: `{record['id']}` difficulty: `{record['difficulty']}`",
                f"  gold: `{preview_text(str(record['gold_answer']), 120)}`",
                f"  pred: `{preview_text(record['extracted_final_answer'], 120)}`",
            ]
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    runtime_cfg = config["runtime"]
    dpo_cfg = copy.deepcopy(config["dpo"])
    paths_cfg = config["paths"]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir = output_dir / "logs"
    logging_dir.mkdir(parents=True, exist_ok=True)

    init_adapter_path = args.init_adapter_path.resolve()
    if not init_adapter_path.exists():
        raise FileNotFoundError(f"Init adapter path does not exist: {init_adapter_path}")

    train_path = resolve_project_path(paths_cfg["dpo_train"])
    eval_path = resolve_project_path(paths_cfg["dpo_val"])
    raw_dataset = load_dataset("json", data_files={"train": str(train_path), "validation": str(eval_path)})
    raw_train_dataset = raw_dataset["train"]
    raw_eval_dataset = raw_dataset["validation"]
    sample_checks = sample_format_checks(raw_train_dataset, args.sample_check_count, runtime_cfg["seed"])

    print(f"train_samples={len(raw_train_dataset)}")
    print(f"val_samples={len(raw_eval_dataset)}")
    print("sample_format_checks:")
    for example in sample_checks:
        print(
            f"  id={example['id']} difficulty={example['difficulty']} "
            f"subset={example['subset']} type={example['question_type']}"
        )
        print(f"  prompt={preview_text(example['prompt'])}")
        print(f"  chosen={preview_text(example['chosen'])}")
        print(f"  rejected={preview_text(example['rejected'])}")

    base_model, tokenizer, load_info = load_model_and_tokenizer(
        config,
        enable_4bit=not args.disable_4bit,
    )
    base_model = prepare_model_for_training(base_model, gradient_checkpointing=True)
    model = PeftModel.from_pretrained(base_model, str(init_adapter_path), is_trainable=True)
    model.config.use_cache = False

    ref_base_model, _, _ = load_model_and_tokenizer(
        config,
        enable_4bit=not args.disable_4bit,
    )
    ref_model = PeftModel.from_pretrained(ref_base_model, str(init_adapter_path), is_trainable=False)
    ref_model.eval()

    train_dataset = format_dpo_split(raw_train_dataset, tokenizer)
    eval_dataset = format_dpo_split(raw_eval_dataset, tokenizer)

    learning_rate = float(args.learning_rate if args.learning_rate is not None else dpo_cfg["learning_rate"])
    num_train_epochs = float(args.num_train_epochs if args.num_train_epochs is not None else dpo_cfg["num_train_epochs"])

    trainer_args = DPOConfig(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        logging_dir=str(logging_dir),
        report_to="none",
        seed=runtime_cfg["seed"],
        data_seed=runtime_cfg["seed"],
        bf16=load_info["precision"] == "bf16",
        fp16=load_info["precision"] == "fp16",
        per_device_train_batch_size=int(dpo_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(dpo_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(dpo_cfg["gradient_accumulation_steps"]),
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        warmup_ratio=float(dpo_cfg["warmup_ratio"]),
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
        max_length=int(dpo_cfg["max_length"]),
        max_prompt_length=int(dpo_cfg["max_prompt_length"]),
        max_completion_length=int(dpo_cfg["max_length"]) - int(dpo_cfg["max_prompt_length"]),
        beta=float(dpo_cfg["beta"]),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        weight_decay=0.0,
        max_grad_norm=1.0,
    )

    training_args_path = output_dir / "training_args_dpo.json"
    training_args_path.write_text(trainer_args.to_json_string(), encoding="utf-8")
    save_json(
        output_dir / "resolved_run_config_dpo.json",
        {
            "base_config": config,
            "cli_args": vars(args),
            "effective_training_args": json.loads(trainer_args.to_json_string()),
            "init_adapter_path": str(init_adapter_path),
        },
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=trainer_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=None,
    )

    resume_checkpoint = resolve_resume_checkpoint(output_dir, args.resume_from_checkpoint)
    policy_summary = trainable_parameter_summary(trainer.model)
    reference_summary = trainable_parameter_summary(ref_model)

    print(f"model_source={load_info['model_source']}")
    print(f"precision={load_info['precision']}")
    print(f"using_4bit={load_info['using_4bit']}")
    print(f"init_adapter_path={init_adapter_path}")
    print(f"output_dir={output_dir}")
    print(f"resume_checkpoint={resume_checkpoint}")
    print(policy_summary)
    print(f"ref::{reference_summary}")

    started_at = datetime.now()
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    final_metrics = dict(train_result.metrics)
    eval_metrics = trainer.evaluate()
    final_metrics.update(eval_metrics)

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)
    trainer.save_state()

    val_preview_path = output_dir / "dpo_val_preview.jsonl"
    val_previews = run_val_preview(
        trainer=trainer,
        tokenizer=tokenizer,
        raw_eval_examples=[raw_eval_dataset[i] for i in range(len(raw_eval_dataset))],
        count=args.val_preview_count,
        output_path=val_preview_path,
        seed=runtime_cfg["seed"] + 1,
    )
    finished_at = datetime.now()

    write_markdown_log(
        path=output_dir / "dpo_train_log.md",
        started_at=started_at,
        finished_at=finished_at,
        output_dir=output_dir,
        train_count=len(raw_train_dataset),
        val_count=len(raw_eval_dataset),
        sample_checks=sample_checks,
        load_info=load_info,
        init_adapter_path=init_adapter_path,
        policy_summary=policy_summary,
        reference_summary=reference_summary,
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
