#!/usr/bin/env python3
"""
Standalone evaluation script for the fine-tuned MedGemma-4B ICD-10 adapter.
Loads the saved adapter from the last checkpoint and runs evaluation on the
held-out eval set without retraining.

Usage:
    python evaluate_finetuned.py
"""

import json
import re
import logging
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ── Configuration ────────────────────────────────────────────────────────
BASE_MODEL = str(Path(__file__).parent / "medgemma-4b-it")
ADAPTER_DIR = str(Path(__file__).parent / "medgemma-icd10-lora-v2" / "checkpoint-epoch-3")
EVAL_DATA = str(Path(__file__).parent / "generated_training_data" / "eval_split.json")
OUTPUT_FILE = str(Path(__file__).parent / "medgemma-icd10-lora-v2" / "eval_results_epoch3.json")

# ── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_model_and_tokenizer():
    """Load base model with 4-bit quantisation and apply the LoRA adapter."""
    logger.info(f"Loading base model from: {BASE_MODEL}")
    logger.info(f"Loading adapter from: {ADAPTER_DIR}")

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

    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()
    logger.info("Model loaded and set to eval mode")
    return model, tokenizer


def evaluate(model, tokenizer, eval_data: list[dict]) -> dict:
    """Run evaluation on held-out data and return metrics."""
    logger.info(f"\n{'=' * 70}")
    logger.info(f"EVALUATION on {len(eval_data)} held-out cases")
    logger.info(f"{'=' * 70}")

    icd_pattern = re.compile(r"[A-Z]\d{2}\.?\d{0,4}")
    results = []

    for i, item in enumerate(eval_data):
        prompt = item["messages"][0]["content"]
        expected_code = item["code"]
        category = item["category"]

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
        has_any_icd = len(found_codes) > 0

        result = {
            "category": category,
            "expected_code": formatted_expected,
            "found_codes": found_codes,
            "exact_match": exact_match,
            "category_match": category_match,
            "has_any_icd": has_any_icd,
            "response_length": len(response),
            "response_preview": response[:300],
        }
        results.append(result)

        status = "EXACT" if exact_match else ("CAT" if category_match else "MISS")
        logger.info(
            f"  [{i + 1:>3d}/{len(eval_data)}] "
            f"ICD: {status:<7s} | "
            f"Expected: {formatted_expected} | "
            f"Found: {', '.join(found_codes[:3]) if found_codes else 'NONE'} | "
            f"Response: {response.strip()[:80]}"
        )

    # Aggregate
    n = len(results)
    metrics = {
        "total": n,
        "icd_exact_match": sum(1 for r in results if r["exact_match"]),
        "icd_category_match": sum(1 for r in results if r["category_match"]),
        "icd_any_code": sum(1 for r in results if r["has_any_icd"]),
        "avg_response_length": sum(r["response_length"] for r in results) / n if n else 0,
    }

    # Summary
    logger.info(f"\n{'─' * 70}")
    logger.info(f"RESULTS SUMMARY")
    logger.info(f"{'─' * 70}")
    logger.info(f"  ICD-10 exact match:       {metrics['icd_exact_match']:>3d}/{n} ({metrics['icd_exact_match']/n*100:.1f}%)")
    logger.info(f"  ICD-10 category match:    {metrics['icd_category_match']:>3d}/{n} ({metrics['icd_category_match']/n*100:.1f}%)")
    logger.info(f"  Has any ICD-10 code:      {metrics['icd_any_code']:>3d}/{n} ({metrics['icd_any_code']/n*100:.1f}%)")
    logger.info(f"  Avg response length:      {metrics['avg_response_length']:.0f} chars")
    logger.info(f"{'─' * 70}\n")

    return {"metrics": metrics, "results": results}


def main():
    if not torch.cuda.is_available():
        logger.error("CUDA not available. Exiting.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"GPU: {gpu_name}, VRAM: {vram_gb:.1f} GB")

    # Load eval data
    logger.info(f"Loading eval data from: {EVAL_DATA}")
    with open(EVAL_DATA, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    logger.info(f"Loaded {len(eval_data)} eval examples")

    # Load model
    model, tokenizer = load_model_and_tokenizer()

    # Run evaluation
    output = evaluate(model, tokenizer, eval_data)

    # Save results
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to: {OUTPUT_FILE}")

    # Print a few sample outputs
    logger.info(f"\n{'=' * 70}")
    logger.info("SAMPLE OUTPUTS (first 5)")
    logger.info(f"{'=' * 70}")
    for r in output["results"][:5]:
        logger.info(f"  Expected: {r['expected_code']:<10s} | Found: {', '.join(r['found_codes'][:3]) or 'NONE':<15s} | {'EXACT' if r['exact_match'] else 'CAT' if r['category_match'] else 'MISS'}")
        logger.info(f"    Response: {r['response_preview'][:120]}")
        logger.info("")

    logger.info("Done!")


if __name__ == "__main__":
    main()
