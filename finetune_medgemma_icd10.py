#!/usr/bin/env python3
"""
Fine-tune MedGemma-4B-IT on ICD-10 code prediction from clinical notes.
Restricted to: Neoplasms (C00-D49), Nervous System (G00-G99),
               Circulatory System (I00-I99), Musculoskeletal (M00-M99).

Optimised for NVIDIA RTX 5070 Ti (16 GB VRAM) with 4-bit QLoRA.

Usage:
    python finetune_medgemma_icd10.py
"""

import os
import re
import sys
import json
import math
import time
import random
import logging
import datetime
from pathlib import Path
from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType

# ============================================================
# Configuration
# ============================================================

@dataclass
class Config:
    """All tuneable knobs in one place."""
    # Model – local folder (no HF download needed)
    model_id: str = str(Path(__file__).parent / "medgemma-4b-it")
    # ICD-10 data
    icd10_file: str = str(
        Path(__file__).parent
        / "ICD_10_data"
        / "april-1-2026-code-descriptions-in-tabular-order"
        / "Code Descriptions"
        / "icd10cm_order_2026.txt"
    )
    # Category filters (start of ICD-10 code)
    categories: dict = field(default_factory=lambda: {
        "Nervous System": {"prefixes": ["G"]},
    })
    # Sampling – use ALL codes per category for maximum coverage
    max_codes_per_category: int = 9999  # effectively no cap
    # Training – tuned for >95% accuracy
    num_train_epochs: int = 5
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    max_seq_length: int = 768
    warmup_ratio: float = 0.05
    logging_steps: int = 10
    save_strategy: str = "epoch"
    bf16: bool = True
    # LoRA – wider rank for better capacity
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: list = field(
        default_factory=lambda: [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    # Paths
    output_dir: str = "./medgemma-icd10-lora-v2"
    seed: int = 42
    # Eval
    eval_samples: int = 50  # held-out test cases
    # Prompt augmentation multiplier – more variety per code
    augmentation_factor: int = 5


CFG = Config()

# ============================================================
# Logging setup
# ============================================================

LOG_DIR = Path(CFG.output_dir) / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"finetune_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ============================================================
# 1. Parse ICD-10 data
# ============================================================

def parse_icd10_file(filepath: str) -> list[dict]:
    """Parse the fixed-width ICD-10-CM order file.

    Returns list of dicts with keys: code, billable, short_desc, long_desc, category.
    """
    logger.info(f"Parsing ICD-10 data from: {filepath}")
    # Fixed-width layout based on CMS specification:
    #   0-4   : order number (5 chars)
    #   6-12  : ICD-10-CM code (7 chars padded)
    #   14    : billable flag (0 = header, 1 = billable)
    #   16-75 : short description (60 chars)
    #   77+   : long description (rest of line)
    pattern = re.compile(
        r"^(\d{5})\s+"        # order number
        r"([A-Z]\S{2,6})\s+"  # code (3-7 chars)
        r"([01])\s+"          # billable flag
        r"(.+?)\s{2,}"        # short description (followed by 2+ spaces)
        r"(.+?)\s*$"          # long description
    )

    entries = []
    with open(filepath, "r", encoding="utf-8") as fh:
        for line in fh:
            m = pattern.match(line)
            if not m:
                continue
            code = m.group(2).strip()
            billable = m.group(3) == "1"
            short_desc = m.group(4).strip()
            long_desc = m.group(5).strip()
            entries.append({
                "code": code,
                "billable": billable,
                "short_desc": short_desc,
                "long_desc": long_desc,
            })
    logger.info(f"Parsed {len(entries)} total ICD-10-CM entries")
    return entries


def filter_categories(entries: list[dict], categories: dict) -> dict[str, list[dict]]:
    """Filter entries to target categories, billable codes only."""
    result = {}
    for cat_name, cat_info in categories.items():
        prefixes = cat_info["prefixes"]
        filtered = []
        for e in entries:
            if not e["billable"]:
                continue
            code = e["code"]
            # For neoplasms D-range, restrict to D00-D49
            if code.startswith("D") and cat_name == "Neoplasms":
                d_num = int(re.match(r"D(\d+)", code).group(1)) if re.match(r"D(\d+)", code) else 999
                if d_num > 49:
                    continue
            if any(code.startswith(p) for p in prefixes):
                e_copy = dict(e)
                e_copy["category"] = cat_name
                filtered.append(e_copy)
        result[cat_name] = filtered
    for cat, codes in result.items():
        logger.info(f"  {cat}: {len(codes)} billable codes")
    return result


def sample_codes(
    cat_codes: dict[str, list[dict]], max_per_cat: int, seed: int
) -> list[dict]:
    """Stratified random sample, capping each category."""
    rng = random.Random(seed)
    sampled = []
    for cat, codes in cat_codes.items():
        if len(codes) > max_per_cat:
            pick = rng.sample(codes, max_per_cat)
        else:
            pick = codes
        sampled.extend(pick)
        logger.info(f"  Sampled {len(pick)} codes from {cat}")
    rng.shuffle(sampled)
    logger.info(f"Total sampled codes: {len(sampled)}")
    return sampled


# ============================================================
# 2. Synthetic clinical-note generator
# ============================================================

# Template pools for generating realistic clinical vignettes
_AGES = list(range(18, 90))
_GENDERS = ["male", "female"]
_DURATIONS = [
    "2 days", "3 days", "5 days", "1 week", "2 weeks", "1 month",
    "3 months", "6 months", "1 year", "several years",
]
_SETTINGS = [
    "presenting to the emergency department",
    "seen in outpatient clinic",
    "admitted to the hospital",
    "referred by primary care physician",
    "follow-up visit",
]

# Category-specific symptom/presentation templates (8+ per category for variety)
_CATEGORY_TEMPLATES = {
    "Neoplasms": [
        "Patient reports unintentional weight loss of {wt}lbs over {dur}, fatigue, and localized pain. "
        "Imaging reveals a mass consistent with {desc}. Biopsy pending.",
        "History of {desc}. Patient presents for staging workup. CT shows {finding}. "
        "Labs: CBC notable for {lab}. Performance status ECOG {ecog}.",
        "Patient diagnosed with {desc} after {dur} of progressive symptoms including {sx}. "
        "Pathology confirms {desc}. Discussed treatment options.",
        "Screening detected an abnormality. Follow-up imaging and biopsy confirm {desc}. "
        "Patient denies pain but reports {sx} for {dur}. Labs show {lab}.",
        "Referred by PCP for evaluation of {sx} lasting {dur}. Physical exam reveals palpable mass. "
        "CT scan findings consistent with {desc}. Tumor markers ordered.",
        "Post-operative follow-up for {desc}. Patient reports {sx}. Imaging shows {finding}. "
        "Labs: {lab}. Will discuss adjuvant therapy options.",
        "Patient with known {desc} presenting for chemotherapy cycle. ECOG {ecog}. "
        "Current symptoms include {sx}. CBC shows {lab}.",
        "New diagnosis of {desc} after workup for {sx} over {dur}. "
        "PET/CT reveals {finding}. MDT discussion planned.",
    ],
    "Nervous System": [
        "Patient presents with {sx} for {dur}. Neurological exam shows {finding}. "
        "MRI brain: {imaging}. Assessment consistent with {desc}.",
        "Chief complaint: {sx}. Duration {dur}. Past medical history significant for {pmh}. "
        "Exam: {finding}. Consistent with {desc}.",
        "Referred for evaluation of {sx}. EEG {eeg}. History of {desc} with {sx} episodes "
        "occurring {freq}.",
        "Patient brought by family for worsening {sx} over {dur}. Cognitive testing shows decline. "
        "MRI: {imaging}. Clinical picture consistent with {desc}.",
        "Follow-up visit for known {desc}. Current medications partially controlling {sx}. "
        "Neuro exam: {finding}. EEG {eeg}.",
        "New onset {sx} with sudden presentation. CT head negative for acute bleed. "
        "MRI {imaging}. Diagnosed with {desc}. PMH: {pmh}.",
        "Pediatric neurology referral for {sx} occurring {freq}. "
        "Development otherwise normal. EEG {eeg}. Working diagnosis: {desc}.",
        "Patient reports progressive {sx} over {dur}, impacting daily activities. "
        "Exam: {finding}. Nerve conduction study ordered. Consistent with {desc}.",
    ],
    "Circulatory System": [
        "Patient presents with {sx} for {dur}. Vitals: BP {bp}, HR {hr}. "
        "ECG: {ecg}. Labs: {lab}. Assessment: {desc}.",
        "Follow-up for {desc}. Current medications: {meds}. BP today {bp}. "
        "Complains of {sx}. Plan to {plan}.",
        "Emergency presentation: {sx} with acute onset. Troponin {trop}. "
        "ECG shows {ecg}. Diagnosis: {desc}. Initiated {plan}.",
        "Routine cardiovascular check-up. Patient on {meds}. BP {bp}, HR {hr}. "
        "Reports mild {sx}. Echo reviewed. Assessment: {desc}.",
        "Post-MI follow-up. Patient recovering well. Vitals: BP {bp}, HR {hr}. "
        "Labs: {lab}. Current diagnosis: {desc}. Continue {meds}.",
        "Ambulance presentation with {sx}. Unstable vitals: BP {bp}, HR {hr}. "
        "Troponin {trop}. ECG: {ecg}. Urgent diagnosis: {desc}.",
        "Patient with chronic {desc}, managed on {meds}. Reports worsening {sx} over {dur}. "
        "BP {bp}. Labs: {lab}. Medication adjustment planned.",
        "Pre-operative cardiac clearance. Known {desc}. BP {bp}, HR {hr}. "
        "ECG: {ecg}. Stress test {lab}. Cleared with monitoring.",
    ],
    "Musculoskeletal": [
        "Patient presents with {sx} in the {loc} for {dur}. ROM {rom}. "
        "X-ray: {xray}. Assessment: {desc}.",
        "Chronic {sx} affecting {loc}. Tried {prev_tx} without adequate relief. "
        "Exam: {finding}. Diagnosis: {desc}. Plan: {plan}.",
        "Work-related injury {dur} ago. {sx} in {loc}. MRI: {mri}. "
        "Consistent with {desc}. Referred to {referral}.",
        "Return visit for {desc}. Symptoms in {loc}: {sx}. Previous treatment with {prev_tx}. "
        "Exam: {finding}. ROM {rom}. Continue {plan}.",
        "Sports injury to {loc} {dur} ago. Persistent {sx}. X-ray: {xray}. "
        "MRI ordered showing {mri}. Assessment: {desc}.",
        "Elderly patient with progressive {sx} in {loc} over {dur}. "
        "Exam: {finding}. X-ray: {xray}. Diagnosis: {desc}. Referred to {referral}.",
        "Post-surgical follow-up for {desc}. {loc} showing {sx}. ROM {rom}. "
        "Wound healing well. Continue {plan}.",
        "Occupational therapy evaluation for {desc}. Patient reports {sx} in {loc}. "
        "Functional assessment limited by pain. ROM {rom}. Plan: {plan}.",
    ],
}

# Fill-in values
_SYMPTOM_POOLS = {
    "Neoplasms": {
        "wt": lambda r: str(r.randint(5, 30)),
        "finding": lambda r: r.choice([
            "irregular mass with enhancement",
            "lesion with satellite nodules",
            "enlarged lymph nodes",
            "heterogeneous mass with calcifications",
        ]),
        "lab": lambda r: r.choice([
            "Hgb 9.8", "WBC 14.2", "platelets 98k", "LDH elevated",
            "anemia Hgb 8.5", "ESR 55",
        ]),
        "ecog": lambda r: str(r.randint(0, 3)),
        "sx": lambda r: r.choice([
            "fatigue and night sweats", "persistent cough and hemoptysis",
            "palpable mass and pain", "unexplained bleeding",
            "bone pain and weakness",
        ]),
    },
    "Nervous System": {
        "sx": lambda r: r.choice([
            "headache and visual changes", "progressive memory loss",
            "seizure episodes", "tremor and gait instability",
            "numbness and tingling", "weakness in extremities",
            "chronic pain and paresthesias",
        ]),
        "finding": lambda r: r.choice([
            "focal neurological deficit", "hyperreflexia",
            "decreased sensation", "abnormal gait",
            "cranial nerve palsy", "muscle wasting",
        ]),
        "imaging": lambda r: r.choice([
            "white matter changes", "no acute abnormality",
            "cortical atrophy", "enhancing lesion",
            "normal study",
        ]),
        "pmh": lambda r: r.choice([
            "hypertension", "diabetes", "prior stroke",
            "family history of neurological disease",
        ]),
        "eeg": lambda r: r.choice([
            "shows epileptiform discharges", "within normal limits",
            "generalized slowing", "focal sharp waves",
        ]),
        "freq": lambda r: r.choice([
            "weekly", "daily", "monthly", "several times per week",
        ]),
    },
    "Circulatory System": {
        "sx": lambda r: r.choice([
            "chest pain radiating to left arm", "dyspnea on exertion",
            "palpitations", "lower extremity edema",
            "syncope", "dizziness and lightheadedness",
        ]),
        "bp": lambda r: f"{r.randint(110, 190)}/{r.randint(60, 110)}",
        "hr": lambda r: str(r.randint(55, 130)),
        "ecg": lambda r: r.choice([
            "normal sinus rhythm", "atrial fibrillation",
            "ST depression", "ST elevation", "left bundle branch block",
            "sinus tachycardia",
        ]),
        "lab": lambda r: r.choice([
            "BNP 450", "troponin 0.04", "lipid panel LDL 185",
            "Cr 1.4, K 5.1", "BNP 1200, Cr 1.8",
        ]),
        "trop": lambda r: r.choice(["0.08", "2.4", "0.12", "negative"]),
        "meds": lambda r: r.choice([
            "metoprolol 50mg, lisinopril 20mg",
            "amlodipine 10mg, atorvastatin 40mg",
            "furosemide 40mg, carvedilol 12.5mg",
            "aspirin 81mg, clopidogrel 75mg",
        ]),
        "plan": lambda r: r.choice([
            "adjust medications", "cardiac catheterization",
            "echocardiogram and follow-up", "admit for monitoring",
        ]),
    },
    "Musculoskeletal": {
        "sx": lambda r: r.choice([
            "pain and stiffness", "swelling and decreased ROM",
            "chronic aching pain", "sharp pain with movement",
            "morning stiffness lasting >1 hour",
        ]),
        "loc": lambda r: r.choice([
            "right knee", "left hip", "lumbar spine", "right shoulder",
            "bilateral hands", "cervical spine", "left ankle",
        ]),
        "rom": lambda r: r.choice([
            "limited by pain", "full but painful",
            "decreased 30%", "severely restricted",
        ]),
        "xray": lambda r: r.choice([
            "joint space narrowing", "no acute fracture",
            "osteophyte formation", "soft tissue swelling",
            "degenerative changes",
        ]),
        "finding": lambda r: r.choice([
            "tenderness to palpation", "crepitus with movement",
            "joint effusion", "muscle spasm", "positive straight leg raise",
        ]),
        "prev_tx": lambda r: r.choice([
            "NSAIDs and physical therapy", "acetaminophen",
            "corticosteroid injection", "rest and ice",
        ]),
        "mri": lambda r: r.choice([
            "disc herniation at L4-L5", "rotator cuff tear",
            "meniscal tear", "ligament sprain",
            "bone marrow edema",
        ]),
        "plan": lambda r: r.choice([
            "physical therapy referral", "orthopedic consult",
            "steroid injection", "NSAID trial and follow-up",
        ]),
        "referral": lambda r: r.choice([
            "orthopedics", "physical therapy", "rheumatology",
            "pain management",
        ]),
    },
}


def _fill_template(template: str, pool: dict, desc: str, dur: str, rng: random.Random) -> str:
    """Replace {placeholders} in a template string."""
    result = template.replace("{desc}", desc).replace("{dur}", dur)
    # Replace remaining placeholders from pool
    for key, fn in pool.items():
        placeholder = "{" + key + "}"
        if placeholder in result:
            result = result.replace(placeholder, fn(rng))
    return result


def generate_clinical_note(entry: dict, rng: random.Random) -> str:
    """Generate a synthetic clinical note for an ICD-10 code."""
    cat = entry["category"]
    desc = entry["long_desc"]
    age = rng.choice(_AGES)
    gender = rng.choice(_GENDERS)
    dur = rng.choice(_DURATIONS)
    setting = rng.choice(_SETTINGS)

    templates = _CATEGORY_TEMPLATES.get(cat, _CATEGORY_TEMPLATES["Nervous System"])
    pool = _SYMPTOM_POOLS.get(cat, _SYMPTOM_POOLS["Nervous System"])

    template = rng.choice(templates)
    body = _fill_template(template, pool, desc, dur, rng)

    note = f"{age}-year-old {gender}, {setting}. {body}"
    return note


def generate_target_response(entry: dict, rng: random.Random) -> str:
    """Generate a concise ICD-10 prediction as the training target."""
    code = entry["code"]
    desc = entry["long_desc"]
    short = entry["short_desc"]

    # Format the code with a dot (e.g. C340 -> C34.0)
    if len(code) > 3:
        formatted_code = code[:3] + "." + code[3:]
    else:
        formatted_code = code

    # Concise, structured output — the model learns ONLY this mapping
    response_variants = [
        f"ICD-10-CM: {formatted_code} - {desc}",
        f"Diagnosis Code: {formatted_code}\nDescription: {desc}",
        f"ICD-10-CM Code: {formatted_code}\nDiagnosis: {desc}\nShort: {short}",
    ]
    return rng.choice(response_variants)


# ============================================================
# 3. Build training & evaluation datasets
# ============================================================

# Prompt variants for augmentation (10 variants for richer training)
_PROMPT_TEMPLATES = [
    "Given the following clinical note, predict the ICD-10-CM diagnosis code:\n\n{note}",
    "What is the correct ICD-10-CM code for this clinical encounter?\n\n{note}",
    "Assign the appropriate ICD-10-CM diagnosis code for this patient:\n\n{note}",
    "Based on the clinical information below, identify the ICD-10-CM code:\n\n{note}",
    "As a medical coder, determine the ICD-10-CM code for this encounter:\n\n{note}",
    "Review this clinical note and provide the ICD-10-CM diagnosis code:\n\n{note}",
    "Identify the primary ICD-10-CM diagnosis code from this clinical note:\n\n{note}",
    "What ICD-10-CM code best describes this patient\'s condition?\n\n{note}",
    "Provide the ICD-10-CM classification code for the following clinical scenario:\n\n{note}",
    "Determine the correct ICD-10-CM billing code for this clinical encounter:\n\n{note}",
]


def build_datasets(
    sampled_codes: list[dict], eval_count: int, augmentation_factor: int, seed: int
) -> tuple[list[dict], list[dict]]:
    """Build chat-format train and eval datasets."""
    rng = random.Random(seed)

    # Split into train and eval
    rng.shuffle(sampled_codes)
    eval_codes = sampled_codes[:eval_count]
    train_codes = sampled_codes[eval_count:]

    logger.info(f"Train codes: {len(train_codes)}, Eval codes: {len(eval_codes)}")

    # Generate training examples
    train_data = []
    for entry in train_codes:
        for aug_idx in range(augmentation_factor):
            clinical_note = generate_clinical_note(entry, rng)
            target = generate_target_response(entry, rng)
            prompt_template = _PROMPT_TEMPLATES[aug_idx % len(_PROMPT_TEMPLATES)]
            prompt = prompt_template.format(note=clinical_note)
            train_data.append({
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": target},
                ],
                "code": entry["code"],
                "category": entry["category"],
            })

    # Generate eval examples (1 per code, using a fixed prompt template)
    eval_data = []
    for entry in eval_codes:
        clinical_note = generate_clinical_note(entry, rng)
        target = generate_target_response(entry, rng)
        prompt = _PROMPT_TEMPLATES[0].format(note=clinical_note)
        eval_data.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": target},
            ],
            "code": entry["code"],
            "category": entry["category"],
            "clinical_note": clinical_note,
            "expected_response": target,
        })

    logger.info(f"Training examples: {len(train_data)} (with augmentation x{augmentation_factor})")
    logger.info(f"Evaluation examples: {len(eval_data)}")
    return train_data, eval_data


# ============================================================
# 4. PyTorch Dataset & Training Loop (no datasets/pyarrow needed)
# ============================================================

class ICD10Dataset(TorchDataset):
    """Tokenised chat-format dataset for causal LM fine-tuning."""

    def __init__(self, data: list[dict], tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        skipped = 0
        for item in data:
            text = tokenizer.apply_chat_template(
                item["messages"], tokenize=False, add_generation_prompt=False
            )
            enc = tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
            # token_type_ids required by Gemma3 during training
            if "token_type_ids" in enc:
                token_type_ids = enc["token_type_ids"].squeeze(0)
            else:
                token_type_ids = torch.zeros_like(input_ids)
            # Labels = input_ids; padding tokens masked with -100
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


def train_model(
    model,
    tokenizer,
    train_data: list[dict],
    cfg: Config,
) -> float:
    """Pure-PyTorch training loop with gradient accumulation & bf16."""

    # Build dataset & dataloader
    dataset = ICD10Dataset(train_data, tokenizer, cfg.max_seq_length)
    dataloader = DataLoader(dataset, batch_size=cfg.per_device_train_batch_size, shuffle=True)

    total_steps_per_epoch = len(dataloader)
    total_optim_steps = (total_steps_per_epoch // cfg.gradient_accumulation_steps) * cfg.num_train_epochs
    warmup_steps = int(total_optim_steps * cfg.warmup_ratio)

    logger.info("=" * 70)
    logger.info("TRAINING STARTED")
    logger.info(f"  Total epochs: {cfg.num_train_epochs}")
    logger.info(f"  Batch size: {cfg.per_device_train_batch_size}")
    logger.info(f"  Gradient accumulation: {cfg.gradient_accumulation_steps}")
    logger.info(f"  Effective batch size: {cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps}")
    logger.info(f"  Learning rate: {cfg.learning_rate}")
    logger.info(f"  Warmup steps: {warmup_steps} ({cfg.warmup_ratio*100:.0f}%)")
    logger.info(f"  Steps/epoch: {total_steps_per_epoch}, Optimiser steps: {total_optim_steps}")
    logger.info(f"  LoRA: r={cfg.lora_r}, alpha={cfg.lora_alpha}, targets={cfg.lora_target_modules}")
    logger.info("=" * 70)

    # Optimiser & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.01)

    # Cosine scheduler with warmup
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_optim_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.bf16)

    model.train()
    global_step = 0
    optim_step = 0
    running_loss = 0.0
    log_loss = 0.0
    log_count = 0
    train_start = time.time()

    for epoch in range(1, cfg.num_train_epochs + 1):
        epoch_start = time.time()
        epoch_loss = 0.0
        logger.info(f"\n{'─' * 50}")
        logger.info(f"EPOCH {epoch}/{cfg.num_train_epochs} STARTED")
        logger.info(f"{'─' * 50}")

        for step, batch in enumerate(dataloader, 1):
            batch = {k: v.to(model.device) for k, v in batch.items()}

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.bf16):
                outputs = model(**batch)
                loss = outputs.loss / cfg.gradient_accumulation_steps

            scaler.scale(loss).backward()
            step_loss = loss.item() * cfg.gradient_accumulation_steps
            epoch_loss += step_loss
            log_loss += step_loss
            log_count += 1
            global_step += 1

            if step % cfg.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                optim_step += 1

            if global_step % cfg.logging_steps == 0:
                avg = log_loss / log_count if log_count else 0
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"  Step {global_step:>5d} (opt {optim_step:>4d}) | "
                    f"Loss: {avg:.4f} | LR: {lr:.2e} | "
                    f"Epoch: {epoch}.{step}/{total_steps_per_epoch}"
                )
                log_loss = 0.0
                log_count = 0

        # End of epoch
        avg_epoch_loss = epoch_loss / total_steps_per_epoch if total_steps_per_epoch else 0
        elapsed = time.time() - epoch_start
        logger.info(f"EPOCH {epoch} COMPLETED in {elapsed:.1f}s | Avg loss: {avg_epoch_loss:.4f}")

        # Save checkpoint
        ckpt_dir = Path(cfg.output_dir) / f"checkpoint-epoch-{epoch}"
        model.save_pretrained(str(ckpt_dir))
        tokenizer.save_pretrained(str(ckpt_dir))
        logger.info(f"  Checkpoint saved to {ckpt_dir}")

    total_elapsed = time.time() - train_start
    final_loss = epoch_loss / total_steps_per_epoch if total_steps_per_epoch else 0
    logger.info("=" * 70)
    logger.info(f"TRAINING COMPLETED in {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
    logger.info(f"  Final loss: {final_loss:.4f} | Total optim steps: {optim_step}")
    logger.info("=" * 70)
    return final_loss


# ============================================================
# 5. Evaluation harness
# ============================================================

def evaluate_model(
    model,
    tokenizer,
    eval_data: list[dict],
    label: str = "Model",
) -> dict:
    """Evaluate model on held-out data. Returns metrics dict."""
    logger.info(f"\n{'=' * 70}")
    logger.info(f"EVALUATION: {label} on {len(eval_data)} held-out cases")
    logger.info(f"{'=' * 70}")

    results = []
    for i, item in enumerate(eval_data):
        prompt = item["messages"][0]["content"]
        expected_code = item["code"]
        category = item["category"]

        # Format code with dot
        if len(expected_code) > 3:
            formatted_expected = expected_code[:3] + "." + expected_code[3:]
        else:
            formatted_expected = expected_code

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

        # Score the output - check for ICD-10 code match
        icd_pattern = re.compile(r"[A-Z]\d{2}\.?\d{0,4}")
        found_codes = icd_pattern.findall(response)
        exact_match = any(
            c.replace(".", "") == expected_code for c in found_codes
        )
        # Category match: first 3 chars match (same ICD-10 category)
        category_match = any(
            c.replace(".", "")[:3] == expected_code[:3] for c in found_codes
        )
        has_any_icd = len(found_codes) > 0

        result = {
            "category": category,
            "expected_code": formatted_expected,
            "found_codes": found_codes,
            "exact_match": exact_match,
            "category_match": category_match,
            "has_any_icd": has_any_icd,
            "response_length": len(response),
            "response_preview": response[:200],
        }
        results.append(result)

        status = "EXACT" if exact_match else ("CAT" if category_match else "MISS")
        logger.info(
            f"  [{i + 1:>3d}/{len(eval_data)}] "
            f"ICD: {status:<7s} | "
            f"Expected: {formatted_expected} | "
            f"Found: {', '.join(found_codes[:3]) if found_codes else 'NONE'} | "
            f"Cat: {category}"
        )

    # Aggregate metrics
    n = len(results)
    metrics = {
        "total": n,
        "icd_exact_match": sum(1 for r in results if r["exact_match"]),
        "icd_category_match": sum(1 for r in results if r["category_match"]),
        "icd_any_code": sum(1 for r in results if r["has_any_icd"]),
        "avg_response_length": sum(r["response_length"] for r in results) / n if n else 0,
    }

    # Per-category breakdown
    cat_metrics = {}
    for cat in CFG.categories:
        cat_results = [r for r in results if r["category"] == cat]
        if cat_results:
            cn = len(cat_results)
            cat_metrics[cat] = {
                "count": cn,
                "exact_match": sum(1 for r in cat_results if r["exact_match"]),
                "category_match": sum(1 for r in cat_results if r["category_match"]),
            }

    metrics["per_category"] = cat_metrics

    # Print summary
    logger.info(f"\n{'─' * 70}")
    logger.info(f"SUMMARY: {label}")
    logger.info(f"{'─' * 70}")
    logger.info(f"  ICD-10 exact match:       {metrics['icd_exact_match']:>3d}/{n} ({metrics['icd_exact_match']/n*100:.1f}%)")
    logger.info(f"  ICD-10 category match:    {metrics['icd_category_match']:>3d}/{n} ({metrics['icd_category_match']/n*100:.1f}%)")
    logger.info(f"  Has any ICD-10 code:      {metrics['icd_any_code']:>3d}/{n} ({metrics['icd_any_code']/n*100:.1f}%)")
    logger.info(f"  Avg response length:      {metrics['avg_response_length']:.0f} chars")

    logger.info(f"\n  Per-category breakdown:")
    for cat, cm in cat_metrics.items():
        logger.info(
            f"    {cat:<25s}: exact={cm['exact_match']}/{cm['count']}  "
            f"cat_match={cm['category_match']}/{cm['count']}"
        )
    logger.info(f"{'─' * 70}\n")

    return metrics


def print_comparison(baseline: dict, finetuned: dict, n: int):
    """Print a side-by-side comparison table."""
    logger.info("\n" + "=" * 78)
    logger.info("BEFORE vs AFTER FINE-TUNING COMPARISON")
    logger.info("=" * 78)
    logger.info(f"{'Metric':<35s} {'Base MedGemma':>14s} {'Fine-tuned':>12s} {'Delta':>10s}")
    logger.info("-" * 78)

    rows = [
        ("ICD-10 exact match", "icd_exact_match"),
        ("ICD-10 category match (3-char)", "icd_category_match"),
        ("Has any ICD-10 code", "icd_any_code"),
    ]
    for label, key in rows:
        bv = baseline[key]
        fv = finetuned[key]
        delta = fv - bv
        sign = "+" if delta >= 0 else ""
        logger.info(
            f"  {label:<33s} {bv:>5d}/{n}  ({bv/n*100:>5.1f}%) "
            f"{fv:>5d}/{n}  ({fv/n*100:>5.1f}%) "
            f"{sign}{delta:>3d} ({sign}{delta/max(bv,1)*100:.0f}%)"
        )

    logger.info(
        f"  {'Avg response length':<33s} "
        f"{baseline['avg_response_length']:>10.0f}     "
        f"{finetuned['avg_response_length']:>8.0f}     "
        f"{finetuned['avg_response_length'] - baseline['avg_response_length']:>+8.0f}"
    )
    logger.info("=" * 78)


# ============================================================
# 6. Main pipeline
# ============================================================

def main():
    start_time = time.time()
    logger.info("=" * 70)
    logger.info("MedGemma-4B ICD-10 Fine-Tuning Pipeline")
    logger.info(f"Started at: {datetime.datetime.now().isoformat()}")
    logger.info("Categories: Nervous System (G00-G99)")
    logger.info("=" * 70)

    # ── Validate environment ─────────────────────────────────────────────
    if not torch.cuda.is_available():
        logger.error("CUDA not available. This script requires a GPU. Exiting.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"GPU: {gpu_name}, VRAM: {vram_gb:.1f} GB")
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"Model path (local): {CFG.model_id}")

    # ── Step 1: Parse and filter ICD-10 data ─────────────────────────────
    logger.info("\n[STEP 1/6] Parsing ICD-10-CM data...")
    all_entries = parse_icd10_file(CFG.icd10_file)
    cat_codes = filter_categories(all_entries, CFG.categories)

    total_billable = sum(len(v) for v in cat_codes.values())
    logger.info(f"Total billable codes in target categories: {total_billable}")

    # ── Step 2: Load pre-generated LLM training data ──────────────────────
    logger.info("\n[STEP 2/6] Loading LLM-generated training and evaluation datasets...")
    llm_train_path = Path("./generated_training_data/train_split.json")
    llm_eval_path = Path("./generated_training_data/eval_split.json")

    if llm_train_path.exists() and llm_eval_path.exists():
        logger.info(f"Loading LLM-generated data from {llm_train_path.parent}")
        with open(llm_train_path, "r", encoding="utf-8") as f:
            train_data = json.load(f)
        with open(llm_eval_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        # Ensure eval entries have expected_response field for evaluation
        for entry in eval_data:
            if "expected_response" not in entry:
                entry["expected_response"] = entry["messages"][1]["content"]
            if "clinical_note" not in entry:
                entry["clinical_note"] = entry["messages"][0]["content"]
        logger.info(f"Loaded {len(train_data)} train, {len(eval_data)} eval from LLM-generated data")
    else:
        logger.info("No LLM-generated data found, falling back to template generation...")
        sampled = sample_codes(cat_codes, CFG.max_codes_per_category, CFG.seed)
        train_data, eval_data = build_datasets(
            sampled, CFG.eval_samples, CFG.augmentation_factor, CFG.seed
        )

    # Save datasets for reproducibility
    data_dir = Path(CFG.output_dir) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dir / "train_data.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    with open(data_dir / "eval_data.json", "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Datasets saved to {data_dir}")

    logger.info(f"Training examples prepared: {len(train_data)}")

    # ── Step 3: Load model with QLoRA ────────────────────────────────────
    logger.info("\n[STEP 3/6] Loading MedGemma-4B-IT with 4-bit quantisation...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(CFG.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        CFG.model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )
    logger.info("Base model loaded successfully")

    # ── Step 4: Baseline evaluation (before fine-tuning) ─────────────────
    logger.info("\n[STEP 4/6] Running BASELINE evaluation (before fine-tuning)...")
    baseline_metrics = evaluate_model(
        model, tokenizer, eval_data, label="BASELINE (pre-fine-tuning)"
    )

    # ── Step 5: Apply LoRA and train ─────────────────────────────────────
    logger.info("\n[STEP 5/6] Applying LoRA adapters and starting training...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=CFG.lora_r,
        lora_alpha=CFG.lora_alpha,
        target_modules=CFG.lora_target_modules,
        lora_dropout=CFG.lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    final_loss = train_model(model, tokenizer, train_data, CFG)

    # Switch back to eval mode (required by Gemma3 to skip token_type_ids check)
    model.eval()

    # ── Step 6: Post-fine-tuning evaluation ──────────────────────────────
    logger.info("\n[STEP 6/6] Running POST-FINE-TUNING evaluation...")
    finetuned_metrics = evaluate_model(
        model, tokenizer, eval_data, label="FINE-TUNED"
    )

    # ── Comparison ───────────────────────────────────────────────────────
    print_comparison(baseline_metrics, finetuned_metrics, len(eval_data))

    # ── Save adapter ─────────────────────────────────────────────────────
    logger.info("\nSaving LoRA adapter...")
    model.save_pretrained(CFG.output_dir)
    tokenizer.save_pretrained(CFG.output_dir)
    logger.info(f"Adapter saved to: {CFG.output_dir}")

    # Save metrics
    metrics_path = Path(CFG.output_dir) / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {"baseline": baseline_metrics, "finetuned": finetuned_metrics},
            f, indent=2,
        )
    logger.info(f"Metrics saved to: {metrics_path}")

    # ── Print sample outputs ─────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("SAMPLE FINE-TUNED OUTPUTS (3 examples)")
    logger.info("=" * 70)
    rng = random.Random(CFG.seed + 999)
    sample_indices = rng.sample(range(len(eval_data)), min(3, len(eval_data)))
    for idx in sample_indices:
        item = eval_data[idx]
        prompt = item["messages"][0]["content"]
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
            out = model.generate(
                input_ids,
                max_new_tokens=128,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)

        code = item["code"]
        if len(code) > 3:
            code = code[:3] + "." + code[3:]
        logger.info(f"\n{'─' * 60}")
        logger.info(f"Category: {item['category']}  |  Expected ICD-10: {code}")
        logger.info(f"Clinical Note: {item['clinical_note'][:150]}...")
        logger.info(f"Model Output: {response.strip()}")
        logger.info(f"{'─' * 60}")

    total_time = time.time() - start_time
    logger.info(f"\nPipeline completed in {total_time:.1f}s ({total_time / 60:.1f} min)")
    logger.info(f"Log file: {log_file}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
