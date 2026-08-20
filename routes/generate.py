import time
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
PATH = "/generate"

SHARED_PROMPT = "In one sentence, what is a data centre for?"
YOUR_PROMPT = "How does a Fortinet firewall differ from a standard router in a data center environment?"

tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True)

def run(prompt):
    ids = tok(prompt, return_tensors="pt")
    t0 = time.perf_counter()
    out = model.generate(**ids, max_new_tokens=40, do_sample=False)
    dt = time.perf_counter() - t0
    n = out.shape[-1] - ids["input_ids"].shape[-1]
    return {"sample": tok.decode(out[0][ids["input_ids"].shape[-1]:],
            skip_special_tokens=True).strip(),
            "seconds": round(dt, 2), "tokens_per_sec": round(n / dt, 1)}

def handle():
    return {"model": MODEL,
            "shared": run(SHARED_PROMPT),
            "yours": run(YOUR_PROMPT)}
