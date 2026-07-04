

# Installation
"""

# 1. Install Unsloth + dependencies
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "xformers<0.0.27" "trl<=0.24.0" peft accelerate bitsandbytes
!pip install "datasets<4.4.0"
!pip install huggingface_hub

# 2. HuggingFace login
from huggingface_hub import login
login()

!pip install transformers peft accelerate bitsandbytes -q

!pip install wandb --upgrade -q

import wandb
wandb.login()

from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-Coder-7B-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

"""# Fine Tuning

load model
"""

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import json

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-Coder-7B-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

"""Add LoRA Adapters"""

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

"""Format Dataset

"""

from google.colab import drive
from datasets import load_dataset, concatenate_datasets, Dataset

drive.mount('/content/drive')

# ── Alpaca prompt ──
alpaca_prompt = """### Instruction:
{}

### Input:
{}

### Response:
{}"""

def format_prompts(examples):
    texts = []
    for inst, inp, out in zip(examples["instruction"], examples["input"], examples["output"]):
        texts.append(alpaca_prompt.format(inst, inp, out) + tokenizer.eos_token)
    return {"text": texts}

# ── Keywords ──
EMBEDDED_KEYWORDS = [
    "arduino", "esp32", "esp8266", "gpio", "pinmode", "digitalwrite",
    "analogread", "analogwrite", "serial", "uart", "i2c", "spi", "pwm",
    "microcontroller", "embedded", "firmware", "interrupt", "timer",
    "sensor", "actuator", "led", "relay", "servo", "stepper", "mqtt",
    "wifi", "bluetooth", "ble", "lcd", "oled", "stm32", "avr", "rtos",
    "circuit", "voltage", "resistor", "capacitor", "transistor", "diode",
    "oscilloscope", "multimeter", "breadboard", "schematic", "pcb"
]

ELECTRONICS_KEYWORDS = [
    "circuit", "voltage", "current", "resistor", "capacitor", "transistor",
    "diode", "ohm", "watt", "ampere", "inductor", "oscillator", "filter",
    "amplifier", "signal", "frequency", "pcb", "schematic", "component",
    "breadboard", "multimeter", "oscilloscope", "kirchhoff", "thevenin"
]

def has_embedded_keywords(text):
    return any(k in text.lower() for k in EMBEDDED_KEYWORDS)

def has_electronics_keywords(text):
    return any(k in text.lower() for k in ELECTRONICS_KEYWORDS)

COLS = ["instruction", "input", "output"]

def keep_cols(ds):
    return ds.select_columns([c for c in COLS if c in ds.column_names])

# 1. YOUR DATA

your_data = load_dataset(
    "json",
    data_files="/content/drive/MyDrive/A.E.G.I.S-V.1/Combined-Data.json",
    split="train"
)
print(f"✅ Your dataset:        {len(your_data)}")

# 2. CodeFeedback

codefeedback = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train")
codefeedback = codefeedback.map(lambda x: {
    "instruction": x.get("query", ""),
    "input": "",
    "output": x.get("answer", ""),
})
codefeedback = codefeedback.filter(
    lambda x: has_embedded_keywords(x["instruction"] + x["output"])
)
codefeedback = codefeedback.select(range(min(2000, len(codefeedback))))
print(f"✅ CodeFeedback:        {len(codefeedback)}")


# 3. OpenCodeInstruct

opencode = load_dataset("nvidia/OpenCodeInstruct", split="train", streaming=True)
opencode_rows = []
for row in opencode:
    if has_embedded_keywords(row.get("instruction", "") + row.get("response", "")):
        opencode_rows.append({
            "instruction": row.get("instruction", ""),
            "input": "",
            "output": row.get("response", ""),
        })
    if len(opencode_rows) >= 1500:
        break
opencode_ds = Dataset.from_list(opencode_rows)
print(f"✅ OpenCodeInstruct:    {len(opencode_ds)}")

# 4. LeetCode

leetcode = load_dataset("newfacade/LeetCodeDataset", split="train")
leetcode = leetcode.map(lambda x: {
    "instruction": x.get("instruction", "") or x.get("question", "") or "",
    "input":       x.get("input", "") or "",
    "output":      x.get("output", "") or x.get("answer", "") or "",
})
leetcode = leetcode.select(range(min(1000, len(leetcode))))
print(f"✅ LeetCode:            {len(leetcode)}")

# 5. Python-codes-25k

pycodes = load_dataset("flytech/python-codes-25k", split="train")
pycodes = pycodes.map(lambda x: {
    "instruction": x.get("instruction", ""),
    "input":       x.get("input", "") or "",
    "output":      x.get("output", ""),
})
pycodes = pycodes.select(range(min(2000, len(pycodes))))
print(f"✅ Python-codes-25k:   {len(pycodes)}")

# 6. CodeAlpaca

codealpaca = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
codealpaca = codealpaca.filter(
    lambda x: has_embedded_keywords(x["instruction"] + x["output"])
)
print(f"✅ CodeAlpaca:          {len(codealpaca)}")

# 7. Electrical Engineering
elec = load_dataset("STEM-AI-mtl/Electrical-engineering", split="train")
elec = elec.map(lambda x: {
    "instruction": x.get("instruction", "") or x.get("question", "") or "",
    "input":       x.get("input", "") or "",
    "output":      x.get("output", "") or x.get("answer", "") or "",
})
elec = elec.filter(
    lambda x: has_electronics_keywords(x["instruction"] + x["output"])
)
print(f"✅ Electrical Eng: {len(elec)}")

# 8. STM32 HAL Dataset
stm32 = load_dataset("MuratKomurcu/stm32-hal-dataset", split="train")
stm32 = stm32.map(lambda x: {
    "instruction": x.get("instruction", "") or x.get("question", "") or "",
    "input":       x.get("input", "") or "",
    "output":      x.get("output", "") or x.get("answer", "") or "",
})
print(f"✅ STM32 HAL: {len(stm32)}")

# MERGE + SHUFFLE + FORMAT

combined = concatenate_datasets([
    keep_cols(your_data),
    keep_cols(codefeedback),
    keep_cols(opencode_ds),
    keep_cols(leetcode),
    keep_cols(pycodes),
    keep_cols(codealpaca),
    keep_cols(elec),
    keep_cols(stm32),
]).shuffle(seed=42)

print(f"\n🔢 Total examples: {len(combined)}")

dataset = combined.map(format_prompts, batched=True)
print("✅ Dataset formatted and ready")

print(len(combined))

"""# Formatted Datasets

"""

from google.colab import drive
from datasets import load_from_disk

drive.mount('/content/drive')

dataset = load_from_disk("/content/drive/MyDrive/A.E.G.I.S-V.1/formatted_dataset")
print(f"✅ Dataset loaded: {len(dataset)} examples")

"""# Train

"""

import os

print(os.listdir("/content/drive/MyDrive/embedded-coder-checkpoints/checkpoint-5000"))

import os
import torch
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments

# Clear cache and set memory config
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

# Configure training arguments using SFTConfig to avoid PicklingError
training_args = SFTConfig(
    dataset_text_field="text",
    max_seq_length=1024,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    max_steps=5000,
    learning_rate=3e-5,
    warmup_steps=150,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    optim="adamw_8bit",
    logging_steps=10,
    save_steps=100,
    save_total_limit=3,
    output_dir="/content/drive/MyDrive/embedded-coder-checkpoints",
    report_to="wandb",
    run_name="AEGIS-V0.6",
)

# Re-initialize trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
)

# Resume training
checkpoint_path = "/content/drive/MyDrive/embedded-coder-checkpoints/checkpoint-5000"
trainer.train(resume_from_checkpoint=checkpoint_path)

"""# Fine Tuned Model"""

from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name= "/content/drive/MyDrive/A.E.G.I.S-V.1/checkpoint-5000",  # your checkpoint path
    max_seq_length=1024,   # must match training!
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)  # faster inference mode

def format_prompt(instruction, input_text=""):
    if input_text:
        return f"""Below is an instruction that describes a task, paired with an input. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""

def ask_aegis(instruction, input_text=""):
    prompt = format_prompt(instruction, input_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )
    return response

# Test
print(ask_aegis(
    instruction="Write code to blink an LED every 500ms",
    input_text="Platform: Arduino Uno, Pin: 13"
))

"""# Test V.0

"""

with open('/content/drive/MyDrive/A.E.G.I.S-V.1/system_prompt.md', 'r') as f:
    SYSTEM_PROMPT = f.read()

print("System prompt loaded successfully.")
print(f"Length: {len(SYSTEM_PROMPT)} characters")
print("---PREVIEW---")
print(SYSTEM_PROMPT[:15000])

from unsloth import FastLanguageModel

# Switch to inference mode
FastLanguageModel.for_inference(model)

# Test prompt
test_prompt = """### Instruction:
Write any 10 assembly code

### Input:

### Response:
"""

inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=300,
    temperature=0.7,
    do_sample=True,
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

questions = [
    "who are you?"

]

# Change these values for each batch
START = 0
END = 1

for i in range(START, END):
    print("=" * 80)
    print(f"QUESTION {i+1}")
    print("=" * 80)
    print(questions[i])
    print("\nANSWER:\n")

    prompt = f"""
You are an embedded systems expert.

Question:
{questions[i]}

Provide a detailed technical answer with code where appropriate.
"""

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.2,
        do_sample=False,
    )

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print("\n\n")

"""# Inference Script"""

from unsloth import FastLanguageModel
from transformers import TextStreamer
import torch

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH       = ("/content/drive/MyDrive/A.E.G.I.S-V.1")  # local path or HF repo
SYSTEM_PROMPT_FILE = "system_prompt.md"
MAX_NEW_TOKENS   = 1024
TEMPERATURE      = 0.7
REPETITION_PENALTY = 1.15
NO_REPEAT_NGRAM  = 4

# ── Load system prompt from file ──────────────────────────────────────────────

def load_system_prompt(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()

# ── Load model ────────────────────────────────────────────────────────────────

def load_model(model_path: str):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=None,           # auto-detect
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer

# ── Format prompt ─────────────────────────────────────────────────────────────

def format_prompt(system_prompt: str, instruction: str, input_text: str = "") -> str:
    """
    Matches the Alpaca format used during training:
    ### Instruction: ...
    ### Input: ...       (optional)
    ### Response:
    """
    if input_text:
        return (
            f"{system_prompt}\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n"
        )
    else:
        return (
            f"{system_prompt}\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n"
        )

# ── Run inference ─────────────────────────────────────────────────────────────

def run(instruction: str, input_text: str = "", stream: bool = True):
    system_prompt = load_system_prompt(SYSTEM_PROMPT_FILE)
    prompt        = format_prompt(system_prompt, instruction, input_text)

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    if stream:
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        _ = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            repetition_penalty=REPETITION_PENALTY,   # kills repetition loops
            no_repeat_ngram_size=NO_REPEAT_NGRAM,    # blocks repeated n-grams
            streamer=streamer,
        )
    else:
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            repetition_penalty=REPETITION_PENALTY,
            no_repeat_ngram_size=NO_REPEAT_NGRAM,
        )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Strip the prompt — return only the model's response
        response = decoded[len(prompt):].strip()
        return response


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    model, tokenizer = load_model(MODEL_PATH)

    # ── Example 1: Code generation ────────────────────────────────────────────
    print("\n" + "="*60)
    print("EXAMPLE 1 — DHT22 on Arduino")
    print("="*60 + "\n")

    run(
        instruction=(
            "Write an Arduino sketch that reads a DHT22 sensor "
            "and sends temperature and humidity over Serial every 2 seconds."
        )
    )

    # ── Example 2: Debugging ──────────────────────────────────────────────────
    print("\n" + "="*60)
    print("EXAMPLE 2 — Debugging")
    print("="*60 + "\n")

    run(
        instruction="An ESP32 keeps rebooting with a watchdog timer error. What are the likely causes and how would you debug it?"
    )

    # ── Example 3: Instruction + Input (code review) ──────────────────────────
    print("\n" + "="*60)
    print("EXAMPLE 3 — Code review with input")
    print("="*60 + "\n")

    run(
        instruction="Review and optimize this code.",
        input_text=(
            "String data = '';\n"
            "while (Serial.available()) { data += (char)Serial.read(); }"
        )
    )

    # ── Example 4: Easter egg ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("EXAMPLE 4 — Easter egg trigger")
    print("="*60 + "\n")

    run(instruction="AEGIS ONLINE")

"""# visualize results

"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Pull training logs ──
log_history = trainer.state.log_history
steps = [x["step"] for x in log_history if "loss" in x]
loss  = [x["loss"] for x in log_history if "loss" in x]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("AEGIS — Training Report", fontsize=18, fontweight="bold", color="#00FFAA")
fig.patch.set_facecolor("#0D0D0D")

for ax in axes.flat:
    ax.set_facecolor("#1A1A1A")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

# ── 1. Training Loss ──
axes[0,0].plot(steps, loss, color="#00FFAA", linewidth=2)
axes[0,0].fill_between(steps, loss, alpha=0.15, color="#00FFAA")
axes[0,0].set_title("Training Loss")
axes[0,0].set_xlabel("Steps")
axes[0,0].set_ylabel("Loss")
axes[0,0].axhline(y=min(loss), color="#FF4444", linestyle="--", alpha=0.6, label=f"Min: {min(loss):.3f}")
axes[0,0].legend(facecolor="#1A1A1A", labelcolor="white")

# ── 2. Loss Smoothed (moving average) ──
window = 5
smoothed = np.convolve(loss, np.ones(window)/window, mode="valid")
smooth_steps = steps[window-1:]
axes[0,1].plot(smooth_steps, smoothed, color="#FFD700", linewidth=2)
axes[0,1].fill_between(smooth_steps, smoothed, alpha=0.15, color="#FFD700")
axes[0,1].set_title("Loss Trend (Smoothed)")
axes[0,1].set_xlabel("Steps")
axes[0,1].set_ylabel("Loss")

# ── 3. Dataset Composition ──
labels   = ["Your Data", "CodeFeedback", "OpenCode",
            "LeetCode", "Python-25k", "CodeAlpaca", "EE", "STM32"]
sizes    = [len(your_data), len(codefeedback), len(opencode_ds),
            len(leetcode), len(pycodes), len(codealpaca), len(elec), len(stm32)]
colors   = ["#00FFAA","#FFD700","#FF6B6B","#4ECDC4","#45B7D1","#96CEB4","#FFEAA7","#DDA0DD"]
axes[1,0].pie(sizes, labels=labels, colors=colors,
              autopct="%1.1f%%", textprops={"color":"white", "fontsize":8},
              wedgeprops={"edgecolor":"#0D0D0D", "linewidth":2})
axes[1,0].set_title("Dataset Composition")

# ── 4. Training Stats Card ──
axes[1,1].axis("off")
stats = [
    ("Model",        "Qwen2.5-Coder-7B"),
    ("Fine-tune",    "LoRA r=16, alpha=16"),
    ("Steps",        f"{steps[-1]}"),
    ("Final Loss",   f"{loss[-1]:.4f}"),
    ("Best Loss",    f"{min(loss):.4f}"),
    ("Total Data",   f"{sum(sizes):,} examples"),
    ("Quantization", "4-bit (BnB)"),
    ("Hardware",     "T4 — 15GB VRAM"),
]
y = 0.95
for label, value in stats:
    axes[1,1].text(0.05, y, f"{label}:", color="#00FFAA",
                   fontsize=11, fontweight="bold", transform=axes[1,1].transAxes)
    axes[1,1].text(0.45, y, value, color="white",
                   fontsize=11, transform=axes[1,1].transAxes)
    y -= 0.11
axes[1,1].set_title("AEGIS — Model Card")

plt.tight_layout()
plt.savefig("AEGIS_training_report.png", dpi=150,
            bbox_inches="tight", facecolor="#0D0D0D")
plt.show()
print("✅ Saved as AEGIS_training_report.png")

"""# Save and Push to HF"""

model.save_pretrained_merged(
    "AEGIS-V.1",
    tokenizer,
    save_method="merged_16bit",
)

model.push_to_hub_merged(
    "C-28/AEGIS-V.1",
    tokenizer,
    save_method="merged_16bit",
)