```
# 需要先安装miniconda环境
conda create -n llm python=3.10 -y
conda activate llm
# 安装依赖
pip install "torch" "transformers>=4.41" "accelerate" "sentencepiece" "safetensors"
# 创建HF缓存目录
mkdir -p $HOME/hf_cache
export HF_HOME=$HOME/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
# 推理脚本文件，第一次运行需要从huggingface拉取模型权重
cat > infer.py <<'PY'
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,   
    device_map="auto",
    trust_remote_code=True,
)

prompt = "Explain what spatial autocorrelation is in GIS, in 5 bullet points."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

print(tokenizer.decode(out[0], skip_special_tokens=True))
PY
#运行推理文件执行推理
python infer.py
```

