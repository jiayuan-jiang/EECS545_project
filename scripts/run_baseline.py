"""
Multi-model baseline evaluation on 4 financial sentiment benchmarks.

Usage:
  python -u run_baseline.py                                      # run all models
  python -u run_baseline.py --models deepseek mistral           # partial names ok
  python -u run_baseline.py --models deepseek-ai/DeepSeek-R1-Distill-Llama-8B
  python -u run_baseline.py --output my_results.json

Results are saved incrementally after each model completes.
Already-completed models are skipped automatically on re-run.
"""

import argparse
import gc
import json
import os
import re
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ── Config ────────────────────────────────────────────────────────────────────
MAX_NEW_TOKENS = 32
MAX_LENGTH     = 2048
BENCHMARK_DIR  = "./benchmark"

ALL_MODELS = [
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]

BENCHMARKS = [
    "benchmark_en_fpb",
    "benchmark_fiqasa",
    "benchmark_sm_bigdata",
    "benchmark_sm_cikm",
]

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--models", nargs="+", default=None,
                    help="Model IDs or partial names to run (default: all)")
parser.add_argument("--output", default="./baseline_results.json",
                    help="Path to results JSON (default: baseline_results.json)")
args = parser.parse_args()

if args.models:
    selected = []
    for query in args.models:
        matches = [m for m in ALL_MODELS if query.lower() in m.lower()]
        if not matches:
            print(f"[warn] no model matched '{query}', skipping", flush=True)
        selected.extend(m for m in matches if m not in selected)
    MODELS = selected
else:
    MODELS = ALL_MODELS

RESULTS_FILE = args.output

# ── Quantization ──────────────────────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# ── Prompt template ───────────────────────────────────────────────────────────
def build_prompt(instruction: str, input_text: str) -> str:
    return (
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{input_text}\n\n"
        f"### Response:\n"
    )

# ── Label extraction ──────────────────────────────────────────────────────────
LABEL_MAP = {
    "positive": "positive", "negative": "negative",
    "neutral":  "neutral",
    "rise": "positive", "fall": "negative",
    "up":   "positive", "down": "negative",
    "bullish": "positive", "bearish": "negative",
}

def extract_label(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>")[-1]
    text = text.strip().lower()
    for word in re.split(r"\s+", text):
        word = re.sub(r"[^a-z]", "", word)
        if word in LABEL_MAP:
            return LABEL_MAP[word]
    return "unknown"

# ── Evaluate one model on one benchmark ──────────────────────────────────────
def evaluate(model, tokenizer, benchmark_name: str) -> dict:
    path = f"{BENCHMARK_DIR}/{benchmark_name}.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    samples = data.get("test", [])
    print(f"  [{benchmark_name}] {len(samples)} samples", flush=True)

    correct, unknown = 0, 0
    predictions = []

    for i, sample in enumerate(samples):
        prompt = build_prompt(sample["instruction"], sample["input"])
        gold   = sample["output"].strip().lower()

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        gen_ids  = out_ids[0][inputs["input_ids"].shape[1]:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        pred     = extract_label(gen_text)

        if pred == gold:
            correct += 1
        if pred == "unknown":
            unknown += 1

        predictions.append({"gold": gold, "pred": pred, "generated": gen_text})

        if (i + 1) % 10 == 0 or (i + 1) == len(samples):
            print(f"    step {i+1}/{len(samples)}  acc: {correct/(i+1):.4f}", flush=True)
            sys.stdout.flush()

    total    = len(samples)
    accuracy = correct / total if total else 0.0
    print(f"    => accuracy: {accuracy:.4f}  ({correct}/{total})  unknown: {unknown}", flush=True)

    return {
        "benchmark":   benchmark_name,
        "total":       total,
        "correct":     correct,
        "unknown":     unknown,
        "accuracy":    round(accuracy, 6),
        "predictions": predictions,
    }

# ── Load existing results (for resume) ───────────────────────────────────────
if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, encoding="utf-8") as f:
        existing = json.load(f)
    all_model_results = existing.get("models", [])
    done_models = {r["model"] for r in all_model_results}
    print(f"Loaded existing results for: {done_models}", flush=True)
else:
    all_model_results = []
    done_models = set()

# ── Incremental save helper ───────────────────────────────────────────────────
BN_SHORT = {
    "benchmark_en_fpb":     "en_fpb",
    "benchmark_fiqasa":     "fiqasa",
    "benchmark_sm_bigdata": "sm_bigdata",
    "benchmark_sm_cikm":    "sm_cikm",
}

def save_results():
    output = {
        "config": {
            "max_length":     MAX_LENGTH,
            "max_new_tokens": MAX_NEW_TOKENS,
            "quantization":   "4-bit NF4 double-quant fp16",
        },
        "models": all_model_results,
    }
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"  => saved to {RESULTS_FILE}", flush=True)

# ── Main loop ─────────────────────────────────────────────────────────────────
for model_id in MODELS:
    if model_id in done_models:
        print(f"\nSkipping {model_id} (already in results)", flush=True)
        continue

    print(f"\n{'='*60}", flush=True)
    print(f"Model: {model_id}", flush=True)
    print(f"{'='*60}", flush=True)

    model_name = model_id.split("/")[-1]
    cache_dir  = f"/tmp/hf_models/{model_name}"

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir=cache_dir,
    )
    model.eval()
    print("Model loaded.\n", flush=True)

    model_results = []
    for bname in BENCHMARKS:
        result = evaluate(model, tokenizer, bname)
        model_results.append(result)

    all_model_results.append({
        "model":    model_id,
        "results":  [
            {k: v for k, v in r.items() if k != "predictions"}
            for r in model_results
        ],
        "detailed": model_results,
    })

    save_results()

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Model unloaded: {model_id}", flush=True)

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'='*75}", flush=True)
print(f"{'Model':<40} {'en_fpb':>8} {'fiqasa':>8} {'sm_bigdata':>10} {'sm_cikm':>8}", flush=True)
print(f"{'-'*75}", flush=True)
for mr in all_model_results:
    name = mr["model"].split("/")[-1]
    accs = {r["benchmark"]: r["accuracy"] for r in mr["results"]}
    row  = f"{name:<40}"
    for bname in BENCHMARKS:
        row += f" {accs.get(bname, 0):>8.4f}"
    print(row, flush=True)
print(f"{'='*75}", flush=True)
print(f"\nFull results saved → {RESULTS_FILE}", flush=True)
