#!/usr/bin/env python3
"""
Gradio app for ICD-10-CM code prediction using fine-tuned MedGemma-4B.
Uses retrieval-augmented generation (RAG): retrieves top candidate ICD-10 codes
via TF-IDF similarity, includes them in the prompt, and lets the model select
the best match. This converts hard recall into easier classification.

Loads the LoRA adapter (v2 checkpoint-epoch-5) trained on Nervous System (G00-G99).

Usage:
    python app_icd10.py
"""

import re
import math
from pathlib import Path
from collections import Counter, defaultdict

import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ── Paths ────────────────────────────────────────────────────────────────
BASE_MODEL = str(Path(__file__).parent / "medgemma-4b-it")
ADAPTER_DIR = str(Path(__file__).parent / "medgemma-icd10-lora-v2" / "checkpoint-epoch-5")
ICD10_FILE = str(
    Path(__file__).parent
    / "ICD_10_data"
    / "april-1-2026-code-descriptions-in-tabular-order"
    / "Code Descriptions"
    / "icd10cm_order_2026.txt"
)

# ── Global state (loaded once) ───────────────────────────────────────────
MODEL = None
TOKENIZER = None
ICD10_INDEX = None  # {"codes": [...], "descriptions": [...], "doc_freqs": Counter, "doc_tokens": [...]}

# Simple English stop words for filtering
_STOP_WORDS = frozenset(
    "a an the is are was were be been being have has had do does did will would "
    "shall should may might can could of in to for on with at by from as into "
    "through during before after above below between out off over under again "
    "further then once here there when where why how all each every both few "
    "more most other some such no nor not only own same so than too very and "
    "but if or because until while".split()
)


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alpha, remove stop words and short tokens."""
    words = re.findall(r"[a-z]{2,}", text.lower())
    return [w for w in words if w not in _STOP_WORDS]


# ── ICD-10 index builder ────────────────────────────────────────────────
def build_icd10_index():
    """Parse G00-G99 billable codes and build a BM25-style index for retrieval."""
    global ICD10_INDEX
    if ICD10_INDEX is not None:
        return

    print("Building ICD-10 code index...")
    pattern = re.compile(
        r"^(\d{5})\s+"
        r"([A-Z]\S{2,6})\s+"
        r"([01])\s+"
        r"(.+?)\s{2,}"
        r"(.+?)\s*$"
    )

    codes = []
    descriptions = []
    doc_tokens = []
    with open(ICD10_FILE, "r", encoding="utf-8") as fh:
        for line in fh:
            m = pattern.match(line)
            if not m:
                continue
            code = m.group(2).strip()
            billable = m.group(3) == "1"
            long_desc = m.group(5).strip()

            if not billable or not code.startswith("G"):
                continue

            formatted = code[:3] + "." + code[3:] if len(code) > 3 else code
            codes.append(formatted)
            descriptions.append(long_desc)
            doc_tokens.append(_tokenize(long_desc))

    # Compute document frequencies for IDF
    doc_freqs = Counter()
    for tokens in doc_tokens:
        for t in set(tokens):
            doc_freqs[t] += 1

    ICD10_INDEX = {
        "codes": codes,
        "descriptions": descriptions,
        "doc_tokens": doc_tokens,
        "doc_freqs": doc_freqs,
        "n_docs": len(codes),
    }
    print(f"Indexed {len(codes)} G-codes for retrieval")


def retrieve_candidates(clinical_note: str, top_k: int = 15) -> list[dict]:
    """Retrieve top-K ICD-10 candidate codes by BM25-style scoring."""
    build_icd10_index()

    query_tokens = _tokenize(clinical_note)
    query_tf = Counter(query_tokens)
    n_docs = ICD10_INDEX["n_docs"]
    doc_freqs = ICD10_INDEX["doc_freqs"]

    # Precompute IDF for query terms
    idf = {}
    for t in query_tf:
        df = doc_freqs.get(t, 0)
        idf[t] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1) if df > 0 else 0

    # BM25 scoring (k1=1.5, b=0.75)
    k1, b = 1.5, 0.75
    avg_dl = sum(len(dt) for dt in ICD10_INDEX["doc_tokens"]) / max(n_docs, 1)

    scores = []
    for doc_idx, doc_toks in enumerate(ICD10_INDEX["doc_tokens"]):
        dl = len(doc_toks)
        doc_tf = Counter(doc_toks)
        score = 0.0
        for t, q_freq in query_tf.items():
            if t not in idf or idf[t] <= 0:
                continue
            tf = doc_tf.get(t, 0)
            tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
            score += idf[t] * tf_norm
        scores.append(score)

    # Get top-K
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    candidates = []
    for idx in top_indices:
        candidates.append({
            "code": ICD10_INDEX["codes"][idx],
            "description": ICD10_INDEX["descriptions"][idx],
            "score": scores[idx],
        })
    return candidates


def load_model():
    """Load base model with 4-bit quantisation + LoRA adapter."""
    global MODEL, TOKENIZER
    if MODEL is not None:
        return

    print("Loading model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    TOKENIZER = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )
    MODEL = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    MODEL.eval()
    print("Model loaded and ready!")


def predict_icd10(clinical_note: str, temperature: float = 0.1, top_k: int = 15) -> str:
    """Given a clinical note, retrieve candidates and predict via constrained decoding."""
    if not clinical_note.strip():
        return "Please enter a clinical note."

    load_model()

    # Step 1: Retrieve candidate codes
    candidates = retrieve_candidates(clinical_note, top_k=top_k)
    candidate_codes = [c["code"] for c in candidates]

    # Step 2: Build RAG prompt with candidates
    candidate_list = "\n".join(
        f"  - {c['code']}: {c['description']}" for c in candidates
    )

    prompt = (
        "Given the following clinical note, predict the most appropriate ICD-10-CM diagnosis code. "
        "Choose from the candidate codes listed below.\n\n"
        f"**Clinical Note:**\n{clinical_note.strip()}\n\n"
        f"**Candidate ICD-10-CM Codes:**\n{candidate_list}\n\n"
        "Respond with ONLY the ICD-10-CM code (e.g. G20.A1). Do not include any other text."
    )

    encoded = TOKENIZER.apply_chat_template(
        [{"role": "user", "content": prompt}],
        return_tensors="pt",
        add_generation_prompt=True,
    )
    if isinstance(encoded, torch.Tensor):
        input_ids = encoded.to(MODEL.device)
    else:
        input_ids = encoded["input_ids"].to(MODEL.device)

    # Step 3: Build trie for constrained decoding
    # Tokenize each candidate code and build a prefix trie of token IDs
    eos_id = TOKENIZER.eos_token_id
    code_token_seqs = []
    for code in candidate_codes:
        # Encode just the code text (no special tokens)
        token_ids = TOKENIZER.encode(code, add_special_tokens=False)
        # Append EOS so the model stops after producing the code
        code_token_seqs.append(token_ids + [eos_id])

    # Build trie: dict of {position: {token_id: set_of_next_token_ids}}
    # trie[depth][token] = set of tokens that can follow at depth+1
    trie = defaultdict(lambda: defaultdict(set))
    max_depth = max(len(seq) for seq in code_token_seqs)
    valid_starts = set()

    for seq in code_token_seqs:
        valid_starts.add(seq[0])
        for depth in range(len(seq) - 1):
            trie[depth][seq[depth]].add(seq[depth + 1])

    # Track generation state per beam (only batch_size=1 here)
    def prefix_allowed_tokens_fn(batch_id: int, sent: torch.Tensor) -> list[int]:
        # How many tokens have been generated so far
        n_generated = sent.shape[0] - input_ids.shape[1]

        if n_generated == 0:
            # First token must be the start of a valid code
            return list(valid_starts)

        if n_generated >= max_depth:
            # Already at max code length, force EOS
            return [eos_id]

        depth = n_generated - 1
        prev_token = sent[-1].item()

        # Look up what can follow this token at this depth
        if depth in trie and prev_token in trie[depth]:
            allowed = list(trie[depth][prev_token])
            return allowed if allowed else [eos_id]
        else:
            # No valid continuation — force EOS
            return [eos_id]

    # Step 4: Generate with constraints
    with torch.no_grad():
        output = MODEL.generate(
            input_ids,
            max_new_tokens=max_depth + 1,
            do_sample=False,  # Greedy for constrained decoding (most reliable)
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            pad_token_id=TOKENIZER.eos_token_id,
        )
    constrained_response = TOKENIZER.decode(
        output[0][input_ids.shape[1]:], skip_special_tokens=True
    ).strip()

    # Also do an unconstrained generation for comparison
    with torch.no_grad():
        output_free = MODEL.generate(
            input_ids,
            max_new_tokens=128,
            temperature=max(temperature, 0.01),
            do_sample=True,
            pad_token_id=TOKENIZER.eos_token_id,
        )
    free_response = TOKENIZER.decode(
        output_free[0][input_ids.shape[1]:], skip_special_tokens=True
    ).strip()

    # Extract ICD-10 codes from both responses
    icd_pattern = re.compile(r"[A-Z]\d{2}\.?\d{0,4}")
    constrained_codes = icd_pattern.findall(constrained_response)
    free_codes = icd_pattern.findall(free_response)

    # Match constrained output to candidate description
    constrained_desc = ""
    for c in candidates:
        if constrained_codes and c["code"].replace(".", "") == constrained_codes[0].replace(".", ""):
            constrained_desc = f" — {c['description']}"
            break

    # Build formatted output
    lines = []
    lines.append("### Constrained Prediction (guaranteed valid code)")
    lines.append(f"**{constrained_response}{constrained_desc}**")
    lines.append("")
    lines.append("### Unconstrained Model Response")
    lines.append(free_response)
    if free_codes:
        lines.append(f"\n*Extracted: {', '.join(free_codes)}*")
    lines.append("")
    lines.append("---")
    lines.append("### Retrieved Candidate Codes (by relevance)")
    lines.append("")
    for i, c in enumerate(candidates[:10], 1):
        is_selected = constrained_codes and c["code"].replace(".", "") == constrained_codes[0].replace(".", "")
        marker = " ✅ **selected**" if is_selected else ""
        lines.append(f"{i}. **{c['code']}** — {c['description']} *(score: {c['score']:.3f})*{marker}")

    return "\n".join(lines)


# ── Example clinical notes (Nervous System G00-G99) ─────────────────────
EXAMPLES = [
    # G20 - Parkinson's disease
    [
        "68-year-old male presenting with progressive resting tremor in the right hand, "
        "bradykinesia, and cogwheel rigidity. Symptoms started 18 months ago and have "
        "gradually worsened. Patient reports difficulty with fine motor tasks and a shuffling "
        "gait. No family history of movement disorders. MRI brain unremarkable. "
        "DaTscan shows reduced dopamine transporter uptake in the left putamen. "
        "Diagnosis: Parkinson's disease.",
        0.1, 15,
    ],
    # G35 - Multiple sclerosis
    [
        "34-year-old female with recurrent episodes of optic neuritis over the past 2 years. "
        "MRI shows multiple periventricular white matter lesions with gadolinium enhancement. "
        "CSF analysis reveals oligoclonal bands. Patient reports numbness and tingling in "
        "bilateral lower extremities, fatigue, and bladder urgency. Visual evoked potentials "
        "are prolonged. Meets McDonald criteria for relapsing-remitting multiple sclerosis.",
        0.1, 15,
    ],
    # G43.909 - Migraine, unspecified
    [
        "28-year-old female with recurring severe headaches occurring 3-4 times per month. "
        "Headaches are unilateral, pulsating, and last 4-72 hours. Associated with nausea, "
        "photophobia, and phonophobia. Patient reports visual aura preceding some episodes "
        "consisting of zigzag lines. No neurological deficits on examination. "
        "CT head normal. Diagnosis: migraine.",
        0.1, 15,
    ],
    # G40 - Epilepsy
    [
        "22-year-old male brought to the ER after witnessed generalized tonic-clonic seizure "
        "lasting approximately 2 minutes. Postictal confusion for 20 minutes. EEG shows "
        "bilateral spike-and-wave discharges at 3 Hz. This is the third unprovoked seizure "
        "in 6 months. MRI brain is normal. No history of head trauma or CNS infection. "
        "Family history positive for epilepsy in maternal uncle.",
        0.1, 15,
    ],
    # G47.33 - Obstructive sleep apnea
    [
        "55-year-old obese male (BMI 38) presenting with excessive daytime somnolence, "
        "loud snoring, and witnessed apneic episodes during sleep. Epworth Sleepiness Scale "
        "score is 16/24. Polysomnography reveals AHI of 32 events/hour with oxygen "
        "desaturations to 78%. Patient reports morning headaches and difficulty concentrating. "
        "Blood pressure 148/92. Diagnosis: obstructive sleep apnea, severe.",
        0.1, 15,
    ],
    # G62.9 - Polyneuropathy
    [
        "62-year-old diabetic patient with progressive numbness and burning pain in bilateral "
        "feet over the past year. Examination reveals decreased sensation to light touch and "
        "pinprick in a stocking distribution, absent ankle reflexes bilaterally, and reduced "
        "vibration sense. Nerve conduction studies show reduced amplitudes and conduction "
        "velocities in distal sensory and motor nerves. HbA1c 8.2%.",
        0.1, 15,
    ],
    # G30 - Alzheimer's disease
    [
        "76-year-old female brought by family for progressive memory loss over 3 years. "
        "MMSE score 18/30. Difficulty with short-term memory, word-finding, and spatial "
        "navigation. Getting lost in familiar places. MRI shows hippocampal atrophy and "
        "generalized cortical thinning. PET scan demonstrates temporoparietal hypometabolism. "
        "CSF biomarkers show elevated phospho-tau and decreased amyloid-beta-42. "
        "Diagnosis: Alzheimer's disease.",
        0.1, 15,
    ],
    # G51.0 - Bell's palsy
    [
        "45-year-old male presenting with acute onset right-sided facial weakness noticed "
        "this morning. Unable to close right eye, raise right eyebrow, or smile on the right "
        "side. No hearing loss, no vesicles in ear canal. Taste sensation reduced on anterior "
        "two-thirds of tongue on the right. No limb weakness or sensory deficits. "
        "CT head unremarkable. Diagnosis: Bell's palsy.",
        0.1, 15,
    ],
    # G25.0 - Essential tremor
    [
        "58-year-old female with bilateral hand tremor that worsens with purposeful movement "
        "such as writing and holding a cup. Tremor has been gradually progressive over 5 years. "
        "Family history: father and sister have similar tremor. Tremor improves with small "
        "amounts of alcohol. No resting tremor, no rigidity, no bradykinesia. "
        "DaTscan normal. Diagnosis: essential tremor.",
        0.1, 15,
    ],
    # G71.0 - Muscular dystrophy
    [
        "8-year-old male with progressive proximal muscle weakness. Uses Gowers' maneuver "
        "to rise from floor. Pseudohypertrophy of calf muscles present. CK level markedly "
        "elevated at 15,000 U/L. EMG shows myopathic pattern. Genetic testing confirms "
        "deletion in dystrophin gene. Maternal uncle died of cardiomyopathy at age 25. "
        "Diagnosis: Duchenne muscular dystrophy.",
        0.1, 15,
    ],
]

# ── Build Gradio interface ───────────────────────────────────────────────
with gr.Blocks(
    title="ICD-10-CM Code Predictor — Nervous System (G00-G99)",
) as demo:
    gr.Markdown(
        """
        # 🧠 ICD-10-CM Code Predictor — Nervous System (RAG-enhanced)
        **Fine-tuned MedGemma-4B** with retrieval-augmented generation on ICD-10-CM codes G00–G99.

        **How it works:**
        1. Your clinical note is matched against ~700 G-code descriptions using TF-IDF similarity
        2. The top candidate codes are included in the prompt
        3. The fine-tuned model selects the best matching ICD-10-CM code

        > **Note:** This is a research demo. Do not use for actual clinical coding decisions.
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            clinical_note_input = gr.Textbox(
                label="Clinical Note",
                placeholder="Enter clinical note here...",
                lines=8,
            )
            with gr.Row():
                temperature_slider = gr.Slider(
                    minimum=0.01,
                    maximum=1.0,
                    value=0.1,
                    step=0.05,
                    label="Temperature",
                )
                topk_slider = gr.Slider(
                    minimum=5,
                    maximum=30,
                    value=15,
                    step=1,
                    label="Candidate codes (top-K)",
                )
            predict_btn = gr.Button("Predict ICD-10 Code", variant="primary")

        with gr.Column(scale=2):
            output_text = gr.Markdown(label="Prediction")

    predict_btn.click(
        fn=predict_icd10,
        inputs=[clinical_note_input, temperature_slider, topk_slider],
        outputs=output_text,
    )
    clinical_note_input.submit(
        fn=predict_icd10,
        inputs=[clinical_note_input, temperature_slider, topk_slider],
        outputs=output_text,
    )

    gr.Examples(
        examples=EXAMPLES,
        inputs=[clinical_note_input, temperature_slider, topk_slider],
        label="Example Clinical Notes (Nervous System Conditions)",
    )

    gr.Markdown(
        """
        ---
        **Model:** MedGemma-4B-IT + QLoRA v2 (checkpoint-epoch-5, LLM-generated data) | **Retrieval:** BM25 over ~665 G-codes
        | **Quantisation:** 4-bit NF4 | **Adapter:** LoRA r=32, α=64
        """
    )


if __name__ == "__main__":
    print("Starting ICD-10 prediction app...")
    print("Model will be loaded on first prediction request.")
    demo.launch(share=False)
