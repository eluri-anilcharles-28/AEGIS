# A.E.G.I.S V-1.0

## Model Information

Model: A.E.G.I.S V1.0
Full Name: Automated Embedded Generative Intelligence System
Base Model: unsloth/Qwen2.5-Coder-7B-bnb-4bit
Method: QLoRA Fine-Tuning via Unsloth
Domain: Embedded Systems Code Generation
Built By: C-28 & A-47

# Training Configuration

- Training Steps: 5,000
- LoRA Rank (r): 16
- LoRA Alpha: 16
- Optimizer: AdamW 8-bit
- Scheduler: Cosine Decay
- Effective Batch Size: 8
- Max Sequence Length (Training): 1024
- Max Sequence Length (Inference): 2048

# Final Metrics

- Final Training Loss: 0.16366977691650392
- Final Epoch: 1.0643
- Final Gradient Norm: 0.073978

# Supported Platforms

- Arduino
- ESP32
- STM32
- AVR Assembly

# Intended Use

A.E.G.I.S is a domain-specialized language model designed for:

- Embedded firmware development
- Microcontroller programming
- Hardware-aware code generation
- Embedded debugging assistance
- Register-level programming guidance

# Limitations

- No validation split
- No quantitative benchmark suite
- Approximately 1.06 training epochs
- May require datasheet verification for hardware-specific values
- Python code is bleeding into it due to more examples of it                                            

# License
   Apache 2.0

# Authors

C-28 [ Eluri Anil Charles ] & A-47 [ Yoganand ]
Independent Researchers
