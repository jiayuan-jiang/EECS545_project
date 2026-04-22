"""
Evaluate fine-tuned DeepSeek-R1-Distill-Llama-8B (LoRA) on 4 benchmarks.

Usage:
  python -u run_finetuned.py 2>&1 | tee run_finetuned.log
"""

import gc
import json
import re
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL    = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
ADAPTER_PATH  = "./models/deepseek-financial-lora-final"
CACHE_DIR     = "/tmp/hf_models/DeepSeek-R1-Distill-Llama-8B"
MAX_NEW_TOKENS = 32
MAX_LENGTH     = 2048
BENCHMARK_DIR  = "./benchmark"
RESULTS_FILE   = "./finetuned_results.json"

BENCHMARKS = [
    "benchmark_en_fpb",
    "benchmark_fiqasa",
    "benchmark_sm_bigdata",
    "benchmark_sm_cikm",
]

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

# ── Evaluate ──────────────────────────────────────────────────────────────────
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

# ── Load model ────────────────────────────────────────────────────────────────
print(f"Loading base model: {BASE_MODEL}", flush=True)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, cache_dir=CACHE_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    cache_dir=CACHE_DIR,
)

print(f"Loading LoRA adapter: {ADAPTER_PATH}", flush=True)
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()
print("Model ready.\n", flush=True)

# ── Run benchmarks ────────────────────────────────────────────────────────────
all_results = []
for bname in BENCHMARKS:
    result = evaluate(model, tokenizer, bname)
    all_results.append(result)

# ── Save ──────────────────────────────────────────────────────────────────────
output = {
    "model":      BASE_MODEL,
    "adapter":    ADAPTER_PATH,
    "config": {
        "max_length":     MAX_LENGTH,
        "max_new_tokens": MAX_NEW_TOKENS,
        "quantization":   "4-bit NF4 double-quant fp16",
    },
    "results": [
        {k: v for k, v in r.items() if k != "predictions"}
        for r in all_results
    ],
    "detailed": all_results,
}

with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}", flush=True)
print(f"{'Benchmark':<25} {'Accuracy':>10} {'Correct':>8} {'Unknown':>8}", flush=True)
print(f"{'-'*60}", flush=True)
for r in all_results:
    print(f"{r['benchmark']:<25} {r['accuracy']:>10.4f} {r['correct']:>8}/{r['total']:<6} {r['unknown']:>8}", flush=True)
print(f"{'='*60}", flush=True)
print(f"\nSaved → {RESULTS_FILE}", flush=True)
