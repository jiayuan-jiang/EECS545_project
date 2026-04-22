"""Quick debug: test each benchmark with correct max_length and print raw model output."""
import json, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto")
model.eval()
print("Model loaded.\n")

BENCHMARKS = [
    ("benchmark_en_fpb",      512),
    ("benchmark_fiqasa",      512),
    ("benchmark_sm_bigdata", 2048),
    ("benchmark_sm_cikm",    2048),
]

for bname, max_len in BENCHMARKS:
    data = json.load(open(f"./benchmark/{bname}.json"))
    samples = data["test"][:3]
    print(f"\n=== {bname} (max_len={max_len}) ===")
    for i, s in enumerate(samples):
        prompt = (
            f"### Instruction:\n{s['instruction']}\n\n"
            f"### Input:\n{s['input']}\n\n"
            f"### Response:\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len).to(model.device)
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id)
        gen = tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"  [{i}] gold={s['output']!r}  output={gen!r}")
