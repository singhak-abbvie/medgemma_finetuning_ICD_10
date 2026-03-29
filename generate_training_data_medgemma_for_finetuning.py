#!/usr/bin/env python3
"""
Generate realistic ICD-10 clinical notes using LOCAL MedGemma-4B-IT.

Replaces the template-based synthetic notes with LLM-generated clinical records
that have natural language, coherent symptom-diagnosis relationships, and diverse
documentation styles.  Runs entirely locally — no API key needed.

Usage:
    python generate_nemo_data.py --pilot          # 50 codes (quick test)
    python generate_nemo_data.py --full            # all ~665 G-codes

    Options:
        --pilot         Generate for 50 codes only (quick validation)
        --full          Generate for all ~665 G-codes (full dataset)
        --augmentation  Notes per code (default: 5)
        --output-dir    Where to save (default: ./generated_training_data)
        --model-path    Path to MedGemma-4B-IT (default: ./medgemma-4b-it)
"""

import re
import sys
import json
import time
import random
import logging
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────
ICD10_FILE = str(
    Path(__file__).parent
    / "ICD_10_data"
    / "april-1-2026-code-descriptions-in-tabular-order"
    / "Code Descriptions"
    / "icd10cm_order_2026.txt"
)

# ── Prompt templates (same as finetune script, for consistent format) ────
_PROMPT_TEMPLATES = [
    "Given the following clinical note, predict the ICD-10-CM diagnosis code:\n\n{note}",
    "What is the correct ICD-10-CM code for this clinical encounter?\n\n{note}",
    "Assign the appropriate ICD-10-CM diagnosis code for this patient:\n\n{note}",
    "Based on the clinical information below, identify the ICD-10-CM code:\n\n{note}",
    "As a medical coder, determine the ICD-10-CM code for this encounter:\n\n{note}",
    "Review this clinical note and provide the ICD-10-CM diagnosis code:\n\n{note}",
    "Identify the primary ICD-10-CM diagnosis code from this clinical note:\n\n{note}",
    "What ICD-10-CM code best describes this patient's condition?\n\n{note}",
    "Provide the ICD-10-CM classification code for the following clinical scenario:\n\n{note}",
    "Determine the correct ICD-10-CM billing code for this clinical encounter:\n\n{note}",
]


# ── ICD-10 parser (same as finetune script) ──────────────────────────────
def parse_g_codes(filepath: str) -> list[dict]:
    """Parse ICD-10-CM file and return billable G-codes."""
    pattern = re.compile(
        r"^(\d{5})\s+"
        r"([A-Z]\S{2,6})\s+"
        r"([01])\s+"
        r"(.+?)\s{2,}"
        r"(.+?)\s*$"
    )
    codes = []
    with open(filepath, "r", encoding="utf-8") as fh:
        for line in fh:
            m = pattern.match(line)
            if not m:
                continue
            code = m.group(2).strip()
            billable = m.group(3) == "1"
            short_desc = m.group(4).strip()
            long_desc = m.group(5).strip()
            if billable and code.startswith("G"):
                codes.append({
                    "code": code,
                    "short_desc": short_desc,
                    "long_desc": long_desc,
                    "category": "Nervous System",
                })
    return codes


def format_code(code: str) -> str:
    """Format code with dot: G43B0 -> G43.B0"""
    return code[:3] + "." + code[3:] if len(code) > 3 else code


def generate_target_response(code: str, long_desc: str, short_desc: str, rng: random.Random) -> str:
    """Generate a concise ICD-10 prediction as the training target."""
    formatted = format_code(code)
    variants = [
        f"ICD-10-CM: {formatted} - {long_desc}",
        f"Diagnosis Code: {formatted}\nDescription: {long_desc}",
        f"ICD-10-CM Code: {formatted}\nDiagnosis: {long_desc}\nShort: {short_desc}",
    ]
    return rng.choice(variants)


# ── Clinical settings & note styles (weighted random choices) ────────────
_CLINICAL_SETTINGS = [
    "emergency department",
    "outpatient neurology clinic",
    "inpatient ward",
    "primary care office",
    "urgent care center",
    "neurology follow-up visit",
    "rehabilitation center",
]
_SETTING_WEIGHTS = [2, 3, 2, 3, 1, 2, 1]

_NOTE_STYLES = [
    "brief assessment note",
    "detailed SOAP-style note",
    "history and physical exam",
    "consultation note",
    "progress note",
]
_STYLE_WEIGHTS = [2, 2, 3, 1, 2]

_SYSTEM_PROMPT = (
    "You are an experienced neurologist writing realistic clinical documentation. "
    "You write in standard medical note format with appropriate clinical detail. "
    "You never mention ICD codes or billing codes in clinical notes. "
    "You describe clinical findings that specifically support the diagnosis "
    "without naming the diagnosis directly."
)


def _build_generation_prompt(
    code: str,
    long_desc: str,
    patient_age: int,
    patient_gender: str,
    clinical_setting: str,
    note_style: str,
) -> str:
    """Build the user prompt that asks the model to write a clinical note."""
    return (
        f"You are an experienced physician writing a clinical note for a "
        f"{patient_age}-year-old {patient_gender} patient seen at a "
        f"{clinical_setting}.\n\n"
        f"The patient has the following ICD-10-CM condition:\n"
        f"Description: {long_desc}\n\n"
        f"Write the note as a {note_style}. Include:\n"
        f"- Chief complaint and history of present illness with realistic timeline\n"
        f"- Relevant review of systems (2-3 pertinent positives and negatives)\n"
        f"- Key physical/neurological exam findings specific to this condition\n"
        f"- Relevant diagnostic results (labs, imaging, EEG, EMG, nerve conduction, etc.)\n"
        f"- Relevant past medical history and medications\n\n"
        f"CRITICAL RULES:\n"
        f"- Do NOT mention any ICD-10 code anywhere in the note\n"
        f"- Do NOT state the exact diagnosis name — describe the clinical presentation "
        f"that would lead a medical coder to assign this specific diagnosis\n"
        f"- Make the symptoms, findings, and test results SPECIFIC enough to distinguish "
        f"this exact condition from similar conditions in the same code family\n"
        f"- Use standard medical abbreviations and clinical documentation style\n"
        f"- Keep the note between 150-400 words\n\n"
        f"Write the clinical note now:"
    )


# ── Local MedGemma generation ────────────────────────────────────────────
def load_model(model_path: str):
    """Load MedGemma-4B-IT with 4-bit quantization."""
    logger.info(f"Loading model from {model_path} (4-bit NF4)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    logger.info("Model loaded.")
    return model, tokenizer


def generate_single_note(
    model,
    tokenizer,
    code: str,
    long_desc: str,
    rng: random.Random,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
) -> dict | None:
    """Generate one clinical note for a given ICD-10 code."""
    patient_age = rng.randint(18, 89)
    patient_gender = rng.choice(["male", "female"])
    clinical_setting = rng.choices(_CLINICAL_SETTINGS, weights=_SETTING_WEIGHTS, k=1)[0]
    note_style = rng.choices(_NOTE_STYLES, weights=_STYLE_WEIGHTS, k=1)[0]

    user_prompt = _build_generation_prompt(
        code, long_desc, patient_age, patient_gender, clinical_setting, note_style,
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True,
    )
    if not isinstance(input_ids, torch.Tensor):
        input_ids = input_ids["input_ids"]
    input_ids = input_ids.to(model.device)
    prompt_len = input_ids.shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
        )

    generated_tokens = output_ids[0][prompt_len:]
    note = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    if len(note) < 50:
        return None

    return {
        "code": code,
        "clinical_note": note,
        "patient_age": str(patient_age),
        "patient_gender": patient_gender,
        "clinical_setting": clinical_setting,
        "note_style": note_style,
    }


def generate_with_local_model(
    model,
    tokenizer,
    g_codes: list[dict],
    augmentation: int,
    output_dir: str,
    seed: int = 42,
    temperature: float = 0.7,
) -> list[dict]:
    """Generate clinical notes for all codes x augmentation using local model."""
    rng = random.Random(seed)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Build shuffled work list
    all_entries = []
    for entry in g_codes:
        for aug_idx in range(augmentation):
            all_entries.append({**entry, "aug_idx": aug_idx})
    rng.shuffle(all_entries)

    total = len(all_entries)
    logger.info(f"Total records to generate: {total}")

    all_generated = []
    checkpoint_interval = 50  # save progress every 50 records
    t0 = time.time()

    for i, entry in enumerate(all_entries):
        code = entry["code"]
        long_desc = entry["long_desc"]

        result = generate_single_note(
            model, tokenizer, code, long_desc, rng, temperature,
        )

        if result is None:
            logger.warning(f"  [{i+1}/{total}] Short/empty note for {code}, retrying...")
            result = generate_single_note(
                model, tokenizer, code, long_desc, rng, temperature=0.9,
            )

        if result is not None:
            all_generated.append(result)

        # Progress logging
        if (i + 1) % 10 == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate if rate > 0 else 0
            logger.info(
                f"  [{i+1}/{total}] generated={len(all_generated)} "
                f"rate={rate:.1f}/s  ETA={eta/60:.1f}min"
            )

        # Periodic checkpoint
        if (i + 1) % checkpoint_interval == 0:
            ckpt_path = out_path / "raw_notes_checkpoint.json"
            with open(str(ckpt_path), "w", encoding="utf-8") as f:
                json.dump(all_generated, f, indent=2, ensure_ascii=False)
            logger.info(f"  Checkpoint saved: {ckpt_path}")

    # Save final raw data
    raw_path = out_path / "raw_notes.json"
    with open(str(raw_path), "w", encoding="utf-8") as f:
        json.dump(all_generated, f, indent=2, ensure_ascii=False)
    logger.info(f"\nRaw notes saved to: {raw_path}")
    logger.info(f"Total generated: {len(all_generated)}")

    return all_generated


def build_training_data(
    generated_notes: list[dict],
    g_code_lookup: dict[str, dict],
    seed: int = 42,
) -> list[dict]:
    """Convert generated notes into training data format matching train_data.json."""
    rng = random.Random(seed)
    train_data = []

    for i, note_entry in enumerate(generated_notes):
        code = note_entry["code"]
        clinical_note = note_entry["clinical_note"]

        # Look up the full code info
        code_info = g_code_lookup.get(code)
        if not code_info:
            logger.warning(f"  Code {code} not found in lookup, skipping")
            continue

        # Generate target response
        target = generate_target_response(
            code, code_info["long_desc"], code_info["short_desc"], rng
        )

        # Pick a prompt template
        prompt_template = _PROMPT_TEMPLATES[i % len(_PROMPT_TEMPLATES)]
        prompt = prompt_template.format(note=clinical_note)

        train_data.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": target},
            ],
            "code": code,
            "category": "Nervous System",
            "clinical_note": clinical_note,
        })

    return train_data


def main():
    parser = argparse.ArgumentParser(description="Generate ICD-10 training data with local MedGemma-4B")
    parser.add_argument("--pilot", action="store_true", help="Pilot mode: 50 codes only")
    parser.add_argument("--full", action="store_true", help="Full mode: all ~665 G-codes")
    parser.add_argument("--augmentation", type=int, default=5, help="Notes per code (default: 5)")
    parser.add_argument("--output-dir", type=str, default="./generated_training_data", help="Output directory")
    parser.add_argument("--model-path", type=str, default="./medgemma-4b-it", help="Path to MedGemma-4B-IT")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature (default: 0.7)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if not args.pilot and not args.full:
        args.pilot = True
        logger.info("No mode specified, defaulting to --pilot (50 codes)")

    logger.info("=" * 70)
    logger.info("Local MedGemma-4B — ICD-10 Clinical Note Generation")
    logger.info(f"Mode: {'PILOT (50 codes)' if args.pilot else 'FULL (~665 codes)'}")
    logger.info(f"Augmentation: {args.augmentation}x per code")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("=" * 70)

    # Parse G-codes
    logger.info("\nParsing ICD-10 G-codes...")
    g_codes = parse_g_codes(ICD10_FILE)
    logger.info(f"Found {len(g_codes)} billable G-codes")

    # Build lookup dict
    g_code_lookup = {e["code"]: e for e in g_codes}

    # Sample if pilot mode
    rng = random.Random(args.seed)
    if args.pilot:
        g_codes = rng.sample(g_codes, min(50, len(g_codes)))
        logger.info(f"Pilot: sampled {len(g_codes)} codes")

    total_records = len(g_codes) * args.augmentation
    logger.info(f"\nWill generate {total_records} clinical notes")

    # Load model
    model, tokenizer = load_model(args.model_path)

    # Generate with local model
    generated = generate_with_local_model(
        model, tokenizer, g_codes, args.augmentation,
        args.output_dir, args.seed, args.temperature,
    )

    # Convert to training format
    logger.info("\nConverting to training data format...")
    train_data = build_training_data(generated, g_code_lookup, args.seed)

    # Save training data (same format as train_data.json)
    out_path = Path(args.output_dir) / "train_data.json"
    with open(str(out_path), "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Training data saved to: {out_path}")
    logger.info(f"Total training examples: {len(train_data)}")

    # Print a few samples for review
    logger.info(f"\n{'=' * 70}")
    logger.info("SAMPLE GENERATED NOTES (first 3)")
    logger.info(f"{'=' * 70}")
    for entry in train_data[:3]:
        code = entry["code"]
        note = entry["clinical_note"]
        logger.info(f"\n  Code: {format_code(code)}")
        logger.info(f"  Note: {note[:300]}...")
        logger.info(f"  Target: {entry['messages'][1]['content']}")
        logger.info(f"{'─' * 50}")

    # Summary stats
    codes_covered = len(set(e["code"] for e in train_data))
    logger.info(f"\nSummary:")
    logger.info(f"  Unique codes covered: {codes_covered}")
    logger.info(f"  Total training examples: {len(train_data)}")
    logger.info(f"  Avg notes per code: {len(train_data) / max(codes_covered, 1):.1f}")
    logger.info(f"\nTo fine-tune with this data:")
    logger.info(f"  1. Copy {out_path} to medgemma-icd10-lora/data/train_data.json")
    logger.info(f"  2. Run: python finetune_medgemma_icd10.py")
    logger.info(f"\nOr to merge with existing template data:")
    logger.info(f"  python -c \"import json; "
                f"a=json.load(open('medgemma-icd10-lora/data/train_data.json')); "
                f"b=json.load(open('{out_path}')); "
                f"json.dump(a+b, open('merged_train_data.json','w'), indent=2)\"")
    logger.info("\nDone!")


if __name__ == "__main__":
    main()
