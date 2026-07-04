

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

!pip install wandb --upgrade -q

import wandb
wandb.login()

from google.colab import drive
drive.mount('/content/drive')

import os

# ── Paths ──────────────────────────────────────────────────────
#CHECKPOINT_PATH = "/content/drive/MyDrive/A.E.G.I.S-V.1/checkpoint-5000"
#SAVE_PATH_LORA  = "/content/drive/MyDrive/A.E.G.I.S-V.1/model-lora"
#SAVE_PATH_MERGED = "/content/drive/MyDrive/A.E.G.I.S-V.1/model-merged"

# Verify checkpoint exists
#if not os.path.exists(CHECKPOINT_PATH):
    #raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}")

#print(f"✅ Checkpoint found: {CHECKPOINT_PATH}")
#print(f"   Contents: {os.listdir(CHECKPOINT_PATH)}")

"""# Load the Base Model"""

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

"""# Load the checkpoint and save"""

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/content/drive/MyDrive/A.E.G.I.S-V.1/checkpoint-5000",
    max_seq_length=2048,
    load_in_4bit=True,
)

from peft import PeftModel
import torch
# Load the adapter weights from your checkpoint
model = PeftModel.from_pretrained( model, CHECKPOINT_PATH, is_trainable=False, )

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/content/drive/MyDrive/A.E.G.I.S-V.1/model-lora",
    max_seq_length=2048,
    load_in_4bit=True,
)

# ═══════════════════════════════════════════════════════════════
# LoRA ONLY (smaller, faster to save)
# This saves just the adapter weights — loads fast, needs base model
# ═══════════════════════════════════════════════════════════════

os.makedirs(SAVE_PATH_LORA, exist_ok=True)

print(f"\n⏳ Saving LoRA adapters to /content/drive/MyDrive/A.E.G.I.S-V.1/ ...")

model.save_pretrained(SAVE_PATH_LORA)
tokenizer.save_pretrained(SAVE_PATH_LORA)

print(f"✅ LoRA model saved → {SAVE_PATH_LORA}")
print(f"   Contents: {os.listdir(SAVE_PATH_LORA)}")

# ═══════════════════════════════════════════════════════════════
# MERGED 16-BIT (for HuggingFace push)
# This merges base + adapters into one model — larger but standalone
# ═══════════════════════════════════════════════════════════════

os.makedirs("/content/drive/MyDrive/A.E.G.I.S-V.1/", exist_ok=True)

print(f"\n⏳ Saving merged 16-bit model to /content/drive/MyDrive/A.E.G.I.S-V.1/..")

model.save_pretrained_merged(
    "/content/drive/MyDrive/A.E.G.I.S-V.1/model-merged",
    tokenizer,
    save_method="merged_16bit",
)

print(f"✅ Merged model saved → /content/drive/MyDrive/A.E.G.I.S-V.1/ ")

# Quick Test

import torch
from transformers import TextStreamer

FastLanguageModel.for_inference(model)

SYSTEM_PROMPT_PATH = "/content/drive/MyDrive/A.E.G.I.S-V.1/system_prompt.md"

if os.path.exists(SYSTEM_PROMPT_PATH):
    with open(SYSTEM_PROMPT_PATH, "r") as f:
        SYSTEM_PROMPT = f.read().strip()
    print(f"✅ System prompt loaded — {len(SYSTEM_PROMPT)} chars")

def ask_aegis(instruction, input_text=""):
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{input_text}\n\n"
        f"### Response:\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.2,
            do_sample=True,
            repetition_penalty=1.15,
            no_repeat_ngram_size=4,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            streamer=streamer,
        )

# print("\n" + "="*60)
# print("INFERENCE TEST — Identity")
# print("="*60)
ask_aegis("who is N-1337?")

"""# Inference"""

import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer
from google.colab import drive

drive.mount('/content/drive')

# Load fine-tuned model at full inference context
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/content/drive/MyDrive/A.E.G.I.S-V.1",
    max_seq_length=2048,               # 1024 was training constraint only
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

# Load system prompt
SYSTEM_PROMPT_PATH = "/content/drive/MyDrive/A.E.G.I.S-V.1/system_prompt.md"
with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read().strip()
print(f"✅ System prompt loaded — {len(SYSTEM_PROMPT)} characters")

# Prompt builder
# NOTE: System prompt prepended directly without ### System: header.
# Model was trained on plain Alpaca format — this preserves that
# structure while injecting behavioral context at inference.
def build_prompt(instruction: str, input_text: str = "") -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{input_text}\n\n"
        f"### Response:\n"
    )

# Inference function
def ask_aegis(
    instruction: str,
    input_text: str = "",
    max_new_tokens: int = 512,
    temperature: float = 0.3,
    stream: bool = True,
) -> str:
    prompt = build_prompt(instruction, input_text)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]

    if input_len > 1800:
        print(f"⚠️  Context: {input_len} tokens — approaching limit")

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        repetition_penalty=1.15,       # fixes repetition loop failure
        no_repeat_ngram_size=4,        # kills repeated phrases
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    if stream:
        streamer = TextStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        model.generate(**gen_kwargs, streamer=streamer)
        return ""
    else:
        with torch.no_grad():
            outputs = model.generate(**gen_kwargs)
        return tokenizer.decode(
            outputs[0][input_len:], skip_special_tokens=True
        ).strip()

"""# Test"""

TESTS = [
    ("Identity",
     "Who are you?", ""),

    ("Easter Egg",
     "AEGIS ONLINE", ""),

    ("Arduino — DHT22",
     "Write an Arduino function to read a DHT22 sensor "
     "and print temperature and humidity over Serial every 2 seconds.", ""),

    ("ESP32 — ADC config",
     "Configure ESP32 ADC on GPIO34 at 12-bit resolution "
     "with 11dB attenuation. Show full initialization code.", ""),

    ("STM32 — UART HAL",
     "Initialize UART2 on STM32F4 at 115200 baud using HAL. "
     "Show clock enable and GPIO alternate function config.", ""),

    ("Debugging — Watchdog",
     "An ESP32 reboots with a watchdog timer error every 30 seconds. "
     "What are the likely causes and how do you debug it?", ""),

    ("Code Review",
     "Review and fix this Arduino code.",
     "String data = '';\n"
     "while (Serial.available()) {\n"
     "    data += (char)Serial.read();\n"
     "}"),

    ("Domain Boundary",
     "Write me a Django REST API endpoint.", ""),

    ("AVR — Register Level",
     "Configure Timer1 on ATmega328P for CTC mode "
     "to generate a 1kHz interrupt. Register-level code only.", ""),

    ("Honesty Check",
     "What is the exact base address of the GPIOA peripheral "
     "on an STM32F103C8T6?", ""),
]

for i, (label, instruction, input_text) in enumerate(TESTS, 1):
    print("\n" + "=" * 60)
    print(f"TEST {i:02d} — {label}")
    print("=" * 60)
    ask_aegis(instruction, input_text)
    print()

"""# Save and Push to HF"""

# Install
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" -q
!pip install --no-deps "trl<=0.24.0" peft accelerate bitsandbytes -q

from google.colab import drive
drive.mount('/content/drive')

from unsloth import FastLanguageModel
from huggingface_hub import login

login()  # paste HF token when prompted

# ═══════════════════════════════════════════════════════════════
# PUSH 1 — LoRA ADAPTERS ONLY
# Small upload (~100MB). Users need base model to run this.
# Repo: eluricharles/AEGIS-V1-LoRA
# ═══════════════════════════════════════════════════════════════

print("⏳ Loading LoRA model...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/content/drive/MyDrive/A.E.G.I.S-V.1/model-lora",
    max_seq_length=2048,
    load_in_4bit=True,
)

print("⏳ Pushing LoRA to HuggingFace...")

model.push_to_hub_merged(
    "C-28/AEGIS",   # ← change to your HF username
    tokenizer,
    save_method="lora",             # adapters only
)

print("✅ LoRA pushed → huggingface.co/eluricharles/AEGIS-V1-LoRA")


# ═══════════════════════════════════════════════════════════════
# PUSH 2 — MERGED 16-BIT
# Large upload (~14GB). Standalone — no base model needed.
# Repo: eluricharles/AEGIS-V1
# ═══════════════════════════════════════════════════════════════

print("\n⏳ Loading merged model...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/content/drive/MyDrive/A.E.G.I.S-V.1/model-merged",
    max_seq_length=2048,
    load_in_4bit=True,
)

print("⏳ Pushing merged model to HuggingFace...")

model.push_to_hub_merged(
    "C-28/AEGIS",        # ← change to your HF username
    tokenizer,
    save_method="merged_16bit",     # full standalone model
)

print("✅ Merged pushed → huggingface.co/eluricharles/AEGIS-V1")
print("\n🎉 Both models live on HuggingFace.")

SAVE_PATH   = "/content/drive/MyDrive/A.E.G.I.S-V.1/merged"
HF_REPO     = "C-28/AEGIS-V1"          # ← update to your HF username

os.makedirs(SAVE_PATH, exist_ok=True)

# Save merged 16-bit weights locally
model.save_pretrained_merged(
    SAVE_PATH,
    tokenizer,
    save_method="merged_16bit",
)
print(f"✅ Merged weights saved → {SAVE_PATH}")

# Push to HuggingFace Hub
model.push_to_hub_merged(
    HF_REPO,
    tokenizer,
    save_method="merged_16bit",
)
print(f"✅ Pushed to HuggingFace → {HF_REPO}")