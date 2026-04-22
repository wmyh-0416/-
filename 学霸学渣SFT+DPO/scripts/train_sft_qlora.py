#!/usr/bin/env python

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

from datasets import load_dataset
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
    parser = argparse.ArgumentParser(description="Reusable SFT QLoRA training skeleton.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--variant",
        choices=("cot", "nocot"),
        default="cot",
        help="Which SFT dataset variant to wire into the trainer.",
    )
    parser.add_argument(
        "--run-train",
        action="store_true",
        help="Actually start training. Omit this flag for dry-run setup only.",
    )
    parser.add_argument(
        "--disable-4bit",
        action="store_true",
        help="Disable 4-bit QLoRA model loading.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    sft_cfg = copy.deepcopy(config["sft"])
    paths_cfg = config["paths"]

    train_path = resolve_project_path(paths_cfg[f"sft_{args.variant}_train"])
    eval_path = resolve_project_path(paths_cfg[f"sft_{args.variant}_val"])
    output_dir = ensure_dir(
        Path(paths_cfg["adapter_root"]) / sft_cfg[f"output_subdir_{args.variant}"]
    )
    logging_dir = ensure_dir(Path(paths_cfg["log_root"]) / f"sft-{args.variant}")

    dataset = load_dataset(
        "json",
        data_files={"train": str(train_path), "validation": str(eval_path)},
    )

    model, tokenizer, load_info = load_model_and_tokenizer(
        config,
        enable_4bit=not args.disable_4bit,
    )
    model = prepare_model_for_training(model, gradient_checkpointing=True)

    trainer_args = SFTConfig(
        output_dir=str(output_dir),
        logging_dir=str(logging_dir),
        report_to="none",
        bf16=load_info["precision"] == "bf16",
        fp16=load_info["precision"] == "fp16",
        per_device_train_batch_size=sft_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=sft_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=sft_cfg["gradient_accumulation_steps"],
        learning_rate=float(sft_cfg["learning_rate"]),
        num_train_epochs=float(sft_cfg["num_train_epochs"]),
        warmup_ratio=float(sft_cfg["warmup_ratio"]),
        logging_steps=int(sft_cfg["logging_steps"]),
        save_steps=int(sft_cfg["save_steps"]),
        eval_steps=int(sft_cfg["eval_steps"]),
        save_strategy="steps",
        eval_strategy="steps",
        max_length=int(sft_cfg["max_length"]),
        packing=bool(sft_cfg["packing"]),
        assistant_only_loss=bool(sft_cfg["assistant_only_loss"]),
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=trainer_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=build_lora_config(config),
    )

    print("SFT trainer ready.")
    print(f"variant={args.variant}")
    print(f"train_samples={len(dataset['train'])}")
    print(f"eval_samples={len(dataset['validation'])}")
    print(f"model_source={load_info['model_source']}")
    print(f"precision={load_info['precision']}")
    print(f"using_4bit={load_info['using_4bit']}")
    print(f"output_dir={output_dir}")
    print(trainable_parameter_summary(trainer.model))

    if not args.run_train:
        print("Dry run only. Pass --run-train to actually start training.")
        return 0

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
