#!/usr/bin/env python3
"""
Resume fine-tuning from the last checkpoint (epoch-5) with a lower learning rate
for additional epochs. Loads the saved LoRA adapter and continues training.

Usage:
    python resume_training.py
"""

import json
import math
import re
import sys
import time
import random
import logging
import datetime
from pathlib import Path

import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ── Configuration ────────────────────────────────────────────────────────
BASE_MODEL = str(Path(__file__).parent / "medgemma-4b-it")
ADAPTER_DIR = str(Path(__file__).parent / "medgemma-icd10-lora-v2" / "checkpoint-epoch-3")
TRAIN_DATA = str(Path(__file__).parent / "generated_training_data" / "train_split.json")
EVAL_DATA = str(Path(__file__).parent / "generated_training_data" / "eval_split.json")
OUTPUT_DIR = str(Path(__file__).parent / "medgemma-icd10-lora-v2")

# Resume training params
ADDITIONAL_EPOCHS = 2          # epochs 4, 5
LEARNING_RATE = 5e-5           # lower LR for continued training
WARMUP_RATIO = 0.1             # longer warmup
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4
MAX_SEQ_LENGTH = 512
LOGGING_STEPS = 10
STARTING_EPOCH = 4             # naming continues from epoch 3
SEED = 42

# ── Logging ──────────────────────────────────────────────────────────────
log_dir = Path(OUTPUT_DIR) / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f"resume_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(log_file), encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ── Dataset (same as main script) ───────────────────────────────────────
class ICD10Dataset(TorchDataset):
    def __init__(self, data: list[dict], tokenizer, max_length: int):
        self.examples = []
        skipped = 0
        for item in data:
            messages = item["messages"]
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            encoded = tokenizer(
                full_text, truncation=True, max_length=max_length,
                padding="max_length", return_tensors="pt",
            )
            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)
            token_type_ids = torch.zeros_like(input_ids)

            if input_ids.shape[0] < 10:
                skipped += 1
                continue

            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            self.examples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "labels": labels,
            })
        logger.info(f"Tokenised {len(self.examples)} examples (skipped {skipped})")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# ── Evaluation ───────────────────────────────────────────────────────────
def evaluate(model, tokenizer, eval_data: list[dict], label: str) -> dict:
    logger.info(f"\n{'=' * 70}")
    logger.info(f"EVALUATION: {label} on {len(eval_data)} held-out cases")
    logger.info(f"{'=' * 70}")

    icd_pattern = re.compile(r"[A-Z]\d{2}\.?\d{0,4}")
    results = []

    for i, item in enumerate(eval_data):
        prompt = item["messages"][0]["content"]
        expected_code = item["code"]

        formatted_expected = (
            expected_code[:3] + "." + expected_code[3:] if len(expected_code) > 3 else expected_code
        )

        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            return_tensors="pt",
            add_generation_prompt=True,
        )
        if isinstance(encoded, torch.Tensor):
            input_ids = encoded.to(model.device)
        else:
            input_ids = encoded["input_ids"].to(model.device)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=128,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)

        found_codes = icd_pattern.findall(response)
        exact_match = any(c.replace(".", "") == expected_code for c in found_codes)
        category_match = any(c.replace(".", "")[:3] == expected_code[:3] for c in found_codes)

        results.append({
            "expected_code": formatted_expected,
            "found_codes": found_codes,
            "exact_match": exact_match,
            "category_match": category_match,
            "response": response.strip()[:200],
        })

        status = "EXACT" if exact_match else ("CAT" if category_match else "MISS")
        logger.info(
            f"  [{i + 1:>3d}/{len(eval_data)}] "
            f"ICD: {status:<7s} | Expected: {formatted_expected} | "
            f"Found: {', '.join(found_codes[:3]) if found_codes else 'NONE'}"
        )

    n = len(results)
    metrics = {
        "total": n,
        "icd_exact_match": sum(1 for r in results if r["exact_match"]),
        "icd_category_match": sum(1 for r in results if r["category_match"]),
    }

    logger.info(f"\n{'─' * 70}")
    logger.info(f"SUMMARY: {label}")
    logger.info(f"{'─' * 70}")
    logger.info(f"  ICD-10 exact match:       {metrics['icd_exact_match']:>3d}/{n} ({metrics['icd_exact_match']/n*100:.1f}%)")
    logger.info(f"  ICD-10 category match:    {metrics['icd_category_match']:>3d}/{n} ({metrics['icd_category_match']/n*100:.1f}%)")
    logger.info(f"{'─' * 70}\n")

    return metrics


def main():
    start_time = time.time()
    logger.info("=" * 70)
    logger.info("RESUME TRAINING — MedGemma-4B ICD-10 (Nervous System G00-G99)")
    logger.info(f"Started at: {datetime.datetime.now().isoformat()}")
    logger.info(f"Resuming from: {ADAPTER_DIR}")
    logger.info(f"Additional epochs: {ADDITIONAL_EPOCHS} (epochs {STARTING_EPOCH}-{STARTING_EPOCH + ADDITIONAL_EPOCHS - 1})")
    logger.info(f"Learning rate: {LEARNING_RATE} (reduced from 1e-4)")
    logger.info("=" * 70)

    if not torch.cuda.is_available():
        logger.error("CUDA not available. Exiting.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"GPU: {gpu_name}, VRAM: {vram_gb:.1f} GB")

    # ── Load data ────────────────────────────────────────────────────────
    logger.info(f"\nLoading training data from: {TRAIN_DATA}")
    with open(TRAIN_DATA, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    logger.info(f"Training examples: {len(train_data)}")

    logger.info(f"Loading eval data from: {EVAL_DATA}")
    with open(EVAL_DATA, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    logger.info(f"Eval examples: {len(eval_data)}")

    # ── Load model + adapter ─────────────────────────────────────────────
    logger.info(f"\nLoading base model: {BASE_MODEL}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )

    logger.info(f"Loading LoRA adapter from: {ADAPTER_DIR}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR, is_trainable=True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    # ── Pre-resume evaluation ────────────────────────────────────────────
    logger.info("\n[PRE-RESUME] Evaluating current checkpoint...")
    model.eval()
    pre_metrics = evaluate(model, tokenizer, eval_data, label="PRE-RESUME (epoch-5)")

    # ── Build dataset & dataloader ───────────────────────────────────────
    dataset = ICD10Dataset(train_data, tokenizer, MAX_SEQ_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    total_steps_per_epoch = len(dataloader)
    total_optim_steps = (total_steps_per_epoch // GRADIENT_ACCUMULATION) * ADDITIONAL_EPOCHS
    warmup_steps = int(total_optim_steps * WARMUP_RATIO)

    logger.info("\n" + "=" * 70)
    logger.info("RESUMED TRAINING")
    logger.info(f"  Epochs: {STARTING_EPOCH}-{STARTING_EPOCH + ADDITIONAL_EPOCHS - 1}")
    logger.info(f"  Batch size: {BATCH_SIZE}")
    logger.info(f"  Gradient accumulation: {GRADIENT_ACCUMULATION}")
    logger.info(f"  Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    logger.info(f"  Learning rate: {LEARNING_RATE}")
    logger.info(f"  Warmup steps: {warmup_steps} ({WARMUP_RATIO*100:.0f}%)")
    logger.info(f"  Steps/epoch: {total_steps_per_epoch}, Optimiser steps: {total_optim_steps}")
    logger.info("=" * 70)

    # ── Optimiser & scheduler ────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_optim_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    # ── Training loop ────────────────────────────────────────────────────
    model.train()
    global_step = 0
    optim_step = 0
    log_loss = 0.0
    log_count = 0

    for epoch_idx in range(ADDITIONAL_EPOCHS):
        epoch_num = STARTING_EPOCH + epoch_idx
        epoch_start = time.time()
        epoch_loss = 0.0

        logger.info(f"\n{'─' * 50}")
        logger.info(f"EPOCH {epoch_num} STARTED")
        logger.info(f"{'─' * 50}")

        for step, batch in enumerate(dataloader, 1):
            batch = {k: v.to(model.device) for k, v in batch.items()}

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss / GRADIENT_ACCUMULATION

            scaler.scale(loss).backward()
            step_loss = loss.item() * GRADIENT_ACCUMULATION
            epoch_loss += step_loss
            log_loss += step_loss
            log_count += 1
            global_step += 1

            if step % GRADIENT_ACCUMULATION == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                optim_step += 1

            if global_step % LOGGING_STEPS == 0:
                avg = log_loss / log_count if log_count else 0
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"  Step {global_step:>5d} (opt {optim_step:>4d}) | "
                    f"Loss: {avg:.4f} | LR: {lr:.2e} | "
                    f"Epoch: {epoch_num}.{step}/{total_steps_per_epoch}"
                )
                log_loss = 0.0
                log_count = 0

        avg_epoch_loss = epoch_loss / total_steps_per_epoch if total_steps_per_epoch else 0
        elapsed = time.time() - epoch_start
        logger.info(f"EPOCH {epoch_num} COMPLETED in {elapsed:.1f}s | Avg loss: {avg_epoch_loss:.4f}")

        # Save checkpoint
        ckpt_dir = Path(OUTPUT_DIR) / f"checkpoint-epoch-{epoch_num}"
        model.save_pretrained(str(ckpt_dir))
        tokenizer.save_pretrained(str(ckpt_dir))
        logger.info(f"  Checkpoint saved to {ckpt_dir}")

    # ── Post-training evaluation ─────────────────────────────────────────
    logger.info("\n[POST-RESUME] Evaluating fine-tuned model...")
    model.eval()
    post_metrics = evaluate(model, tokenizer, eval_data, label="POST-RESUME (epoch-8)")

    # ── Comparison ───────────────────────────────────────────────────────
    n = len(eval_data)
    logger.info("\n" + "=" * 70)
    logger.info("BEFORE vs AFTER RESUME COMPARISON")
    logger.info("=" * 70)
    logger.info(f"{'Metric':<30s} {'Epoch-5':>12s} {'Epoch-8':>12s} {'Delta':>10s}")
    logger.info("-" * 70)
    for label, key in [("ICD-10 exact match", "icd_exact_match"), ("ICD-10 category match", "icd_category_match")]:
        bv = pre_metrics[key]
        fv = post_metrics[key]
        delta = fv - bv
        sign = "+" if delta >= 0 else ""
        logger.info(
            f"  {label:<28s} {bv:>4d}/{n} ({bv/n*100:>5.1f}%) "
            f"{fv:>4d}/{n} ({fv/n*100:>5.1f}%) "
            f"{sign}{delta:>3d}"
        )
    logger.info("=" * 70)

    # ── Save final metrics ───────────────────────────────────────────────
    metrics_path = Path(OUTPUT_DIR) / "resume_metrics.json"
    with open(str(metrics_path), "w", encoding="utf-8") as f:
        json.dump({"pre_resume": pre_metrics, "post_resume": post_metrics}, f, indent=2)
    logger.info(f"Metrics saved to: {metrics_path}")

    total_time = time.time() - start_time
    logger.info(f"\nResume training completed in {total_time:.1f}s ({total_time / 60:.1f} min)")
    logger.info(f"Log file: {log_file}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
