
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
