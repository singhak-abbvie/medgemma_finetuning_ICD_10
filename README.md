# Fine-Tuning MedGemma-4B for ICD-10 Diagnosis Coding

Fine-tune Google's [MedGemma-4B-IT](https://huggingface.co/google/medgemma-4b-it) to predict ICD-10-CM diagnosis codes from clinical notes using QLoRA on a single consumer GPU.

![Python](https://img.shields.io/badge/Python-3.12+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

This project demonstrates an end-to-end pipeline for medical AI:

1. **Synthetic Data Generation** — Use MedGemma itself to generate realistic clinical notes for each ICD-10 code (self-distillation)
2. **QLoRA Fine-Tuning** — 4-bit quantized LoRA training on a consumer GPU (RTX 5070, 12GB VRAM)
3. **Evaluation** — Multi-level metrics: exact code match, category match, and valid code detection
4. **Inference App** — Gradio web interface with BM25 retrieval-augmented generation (RAG) and trie-based constrained decoding

### Results

| Metric | Baseline (No Fine-Tune) | After Fine-Tuning |
|--------|-------------------------|-------------------|
| Exact Code Match | 0% | 38% |
| Category Match (3-char) | 10% | 88% |
| Produces Valid ICD Code | 20% | 100% |

Currently focused on **Chapter 6: Diseases of the Nervous System (G00-G99)** — approximately 665 billable codes.

---

## Project Structure

```
├── finetune_medgemma_icd10.py                          # Main fine-tuning script
├── evaluate_finetuned.py                               # Evaluation script
├── generate_training_data_medgemma_for_finetuning.py   # LLM-based synthetic data generation
├── resume_training.py                                  # Resume training from checkpoint
├── split_data.py                                       # Train/eval data splitter
├── app_icd10.py                                        # Gradio web app (BM25 RAG + constrained decoding)
├── blog_finetuning_medgemma_icd10.md                   # Detailed blog post about the journey
├── .env.example                                        # Environment variable template
├── .gitignore
├── ICD_10_data/                                        # ICD-10-CM source data (CMS)
│   └── april-1-2026-code-descriptions-in-tabular-order/
│       └── Code Descriptions/
│           └── icd10cm_order_2026.txt
├── generated_training_data/                            # Generated synthetic clinical notes
│   ├── raw_notes.json                                  # All generated notes with metadata
│   ├── train_data.json                                 # Training split (formatted for fine-tuning)
│   ├── eval_data.json                                  # Evaluation split
│   └── generation_stats.json                           # Generation run statistics
├── medgemma-4b-it/                                     # Base model weights (download separately)
├── medgemma-icd10-lora/                                # V1 checkpoints (template-based data)
└── medgemma-icd10-lora-v2/                             # V2 checkpoints (LLM-generated data)
```

---

## Prerequisites

- **GPU**: NVIDIA GPU with ≥12GB VRAM (tested on RTX 5070)
- **CUDA**: 12.x
- **Python**: 3.12+
- **Hugging Face account**: With access to [MedGemma-4B-IT](https://huggingface.co/google/medgemma-4b-it) (requires signing the model license)

---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/fine_tuning_ICD_10_UMLS.git
cd fine_tuning_ICD_10_UMLS
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install transformers accelerate bitsandbytes peft gradio python-dotenv
```

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env with your actual tokens:
#   HF_TOKEN=hf_your_token_here
```

### 5. Download Model Weights

```bash
# Login to Hugging Face (requires accepted model license)
huggingface-cli login --token YOUR_HF_TOKEN

# Download MedGemma-4B-IT
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model_id = 'google/medgemma-4b-it'
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained('medgemma-4b-it')
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype='auto')
model.save_pretrained('medgemma-4b-it')
"
```

### 6. Download ICD-10 Data

Download the ICD-10-CM 2026 code descriptions from [CMS.gov](https://www.cms.gov/medicare/coding-billing/icd-10-codes) and place the `icd10cm_order_2026.txt` file in:
```
ICD_10_data/april-1-2026-code-descriptions-in-tabular-order/Code Descriptions/
```

---

## Usage

### Step 1: Generate Synthetic Training Data

Generate realistic clinical notes using MedGemma for self-distillation:

```bash
# Pilot run (50 codes, quick test)
python generate_training_data_medgemma_for_finetuning.py --pilot --augmentation 5

# Full run (all ~665 codes, ~25 hours on RTX 5070)
python generate_training_data_medgemma_for_finetuning.py --full --augmentation 5
```

This generates:
- `generated_training_data/raw_notes.json` — All clinical notes with metadata
- `generated_training_data/train_data.json` — Formatted training data
- `generated_training_data/eval_data.json` — Held-out evaluation data

The script supports **checkpoint/resume** — if interrupted, just re-run the same command.

### Step 2: Fine-Tune the Model

```bash
python finetune_medgemma_icd10.py
```

Trains for 5 epochs with QLoRA. Checkpoints saved to `medgemma-icd10-lora-v2/checkpoint-epoch-{N}/`.

**QLoRA Configuration:**
- 4-bit NF4 quantization with double quantization
- LoRA rank=32, alpha=64, dropout=0.05
- Targets all attention + FFN projections (q, k, v, o, gate, up, down)
- Effective batch size: 8 (batch=2 × grad_accum=4)
- Cosine LR schedule with warmup

### Step 3: Evaluate

```bash
python evaluate_finetuned.py
```

Reports three metrics:
- **Exact Match**: Full ICD-10 code matches ground truth
- **Category Match**: First 3 characters match (e.g., G20 vs G20)
- **Has Any ICD**: Model produced a syntactically valid ICD-10 code

### Step 4: Resume Training (Optional)

To continue training from a checkpoint with a lower learning rate:

```bash
python resume_training.py
```

### Step 5: Launch the Gradio App

```bash
python app_icd10.py
```

Opens a web interface at `http://localhost:7860` where you can paste clinical notes and get ICD-10 predictions.

---

## How It Works

### Data Generation (Self-Distillation)

Rather than manually writing clinical notes, we use MedGemma-4B-IT to generate them — a form of **self-distillation**. For each ICD-10 code, the model generates multiple clinical notes in varied styles:

- SOAP notes
- History & Physical exams
- Progress notes
- Consultation reports
- Discharge summaries
- Brief assessments

10 different prompt templates with randomized patient demographics (age, gender) and clinical settings (ED, outpatient, inpatient, primary care, urgent care) ensure diversity.

### QLoRA Fine-Tuning

We use a **pure PyTorch training loop** (no Hugging Face Trainer dependency) with:

- **4-bit NF4 quantization**: Compresses model weights to ~2.5GB
- **LoRA adapters**: Only ~3% of parameters are trainable
- **Mixed precision (bfloat16)**: Faster compute with minimal quality loss
- **Gradient accumulation**: Simulates larger batch sizes within VRAM limits

### Inference Pipeline

The Gradio app uses a two-stage approach:

1. **BM25 Retrieval**: Scores all ICD-10 codes against the clinical note using BM25 (implemented in pure Python). Returns top-15 candidate codes.

2. **Trie-Based Constrained Decoding**: A prefix trie built from all valid ICD-10 codes restricts token generation at each step. The model can only output tokens that continue a valid code path. This **guarantees 100% valid output** — the model cannot hallucinate non-existent codes.

```
Clinical Note → [BM25 Retrieval] → Top 15 Candidates → [LLM + Trie Constraint] → Valid ICD-10 Code
```

---

## Key Technical Decisions

| Decision | Why |
|----------|-----|
| **Pure PyTorch loop** | Avoids pyarrow/datasets DLL conflicts on Windows |
| **Pure Python BM25** | scikit-learn/scipy had Python 3.14 compatibility issues |
| **Self-distillation** | Free, high-quality, diverse training data without external APIs |
| **All 7 LoRA target modules** | More adaptation capacity for domain-specific tasks |
| **Trie-constrained decoding** | Medical coding demands 100% valid outputs |
| **Per-epoch checkpointing** | V1 showed performance degradation after epoch 5 |

---

## Customization

### Train on Different ICD-10 Chapters

Edit the `TARGET_CATEGORIES` in `finetune_medgemma_icd10.py`:

```python
# Nervous System only (default)
TARGET_CATEGORIES = [("G", "Diseases of the Nervous System")]

# Multiple chapters
TARGET_CATEGORIES = [
    ("G", "Diseases of the Nervous System"),
    ("I", "Diseases of the Circulatory System"),
    ("C", "Neoplasms"),
    ("M", "Diseases of the Musculoskeletal System"),
]
```

### Adjust Training Hyperparameters

Key parameters in `finetune_medgemma_icd10.py`:

```python
LORA_R = 32              # LoRA rank (higher = more capacity, more VRAM)
LORA_ALPHA = 64          # Scaling factor (typically 2× rank)
LEARNING_RATE = 1e-4     # Start here, reduce for resume training
NUM_EPOCHS = 5           # Monitor eval metrics per epoch
MAX_SEQ_LENGTH = 768     # Increase for longer clinical notes
BATCH_SIZE = 2           # Reduce to 1 if running out of VRAM
GRAD_ACCUM_STEPS = 4     # Increase to simulate larger batches
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `token_type_ids is not supported` | Call `model.eval()` before inference |
| CUDA out of memory | Reduce `BATCH_SIZE` to 1, reduce `MAX_SEQ_LENGTH` |
| pyarrow DLL errors | This project uses pure PyTorch — no pyarrow needed |
| Model generates text instead of codes | Ensure you're loading the fine-tuned adapter, not the base model |
| Training loss not decreasing | Check learning rate; try 1e-4 for initial training |

---

## License

This project is for research and educational purposes. 

- **MedGemma model**: Subject to [Google's Gemma license](https://ai.google.dev/gemma/terms)
- **ICD-10-CM data**: Public domain, provided by [CMS.gov](https://www.cms.gov/)
- **Project code**: MIT License

---

## Disclaimer

This is a research project and is **not intended for clinical use** without proper validation, testing, and regulatory approval. ICD-10 coding in production requires certified medical coders and validated systems. Always consult with healthcare professionals for medical coding decisions.
