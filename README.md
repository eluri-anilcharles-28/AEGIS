# AEGIS
A domain-specialized 7B code model for embedded systems engineers. Built on free hardware by independent researchers.

# A.E.G.I.S — Automated Embedded Generative Intelligence System

> *A domain-specialized 7B code model for embedded systems engineers. Built on free hardware by independent researchers.*

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Model](https://img.shields.io/badge/Model-HuggingFace-yellow)](https://huggingface.co/eluricharles/AEGIS-V1)
[![W&B](https://img.shields.io/badge/Tracked-Weights%20%26%20Biases-orange)](https://wandb.ai/eluricharles-independent-researcher)
[![Zenodo](https://img.shields.io/badge/Paper-zenodo-blue)](https://zenodo.org/records/20668166)
[![Python](https://img.shields.io/badge/Python-3.12-blue)]()
[![CUDA](https://img.shields.io/badge/CUDA-12.8-green)]()

---

## What Is A.E.G.I.S?
A.E.G.I.S — Automated Embedded Generative Intelligence System. A General-purpose code models fail at embedded systems. They suggest `malloc` on a 2KB SRAM device. They produce recursive algorithms on platforms with no call stack budget. They give you Linux `/dev/ttyUSB0` code when you asked about a microcontroller UART peripheral.

A.E.G.I.S was built to fix that.

It is a 7B parameter language model fine-tuned specifically for embedded systems development using QLoRA on a single NVIDIA T4 GPU — Google Colab free tier. It understands registers, hardware constraints, deterministic timing, and the low-level reasoning that general models get wrong.

**Builders:** C-28 & A-47 — Independent Researchers
**Base model:** `unsloth/Qwen2.5-Coder-7B-bnb-4bit`
**Training:** 12 runs, 6 version checkpoints, ~14 hours, 5,000 steps
**Final loss:** 0.16366977691650392
**Grad_norm:**	0.07397811114788055
**Learning_rate:** -> 0-1k - 5e-5
                   -> 1k-2k - 2e-5
                   -> 2k-5k - 3e-5
 

---

## Supported Domains

| Platform | Coverage |
|---|---|
| Arduino (AVR) | GPIO, timers, interrupts, I2C, SPI, UART, PWM |
| ESP32 | WiFi, BLE, ADC, DAC, FreeRTOS, deep sleep, MQTT |
| STM32 (HAL) | Peripheral init, DMA, CubeMX patterns, clock config |
| AVR Assembly | Direct register manipulation, ISR, timing |
| Sensor Integration | DHT22, MPU6050, DS18B20, RFID, ultrasonic, LM35 |
| General Embedded | State machines, debouncing, power management |
| Python |

---
## Model Card

| Parameter | Value |
|---|---|
| Base model | `unsloth/Qwen2.5-Coder-7B-bnb-4bit` |
| Total parameters | ~7 billion |
| Fine-tuning method | QLoRA |
| LoRA rank (r) | 16 |
| LoRA alpha | 16 |
| Alpha/r ratio | 1.0 (unit scaling) |
| LoRA dropout | 0 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Trainable parameters | ~802,816 (~0.011% of total) |
| Training steps | 5,000 |
| Effective batch size | 8 (batch=1 × grad_accum=8) |
| Approx. epochs | ~1.06 |
| Max seq length (train) | 1,024 |
| Max seq length (infer) | 2,048 |
| Final training loss | 0.1637 |
| Grad norm (final) | 0.0740 |
| Hardware | NVIDIA T4 16GB (Google Colab free tier) |
| Training duration | ~13–14 hours across 12 runs |
| Dataset size | ~37,000 samples |

---

## Training — Staged Learning Rate

A key methodological contribution is the non-monotonic staged learning rate schedule:

```
Steps    0  – 1,000:  lr = 5e-5   ← peak — aggressive early domain acquisition
Steps 1,000 – 2,000: lr = 2e-5   ← pullback — consolidate weight updates
Steps 2,000 – 5,000: lr = 3e-5   ← recovery — steady domain convergence
```

Each stage uses cosine decay internally. This peak-pullback-recovery pattern differs from standard cosine warmup schedules and was developed through 12 iterative training runs.

---

## Dataset

All sources are open-licensed. Full attribution in `dataset/README.md`.

**Layer 1 — Reasoning Foundation** (preserves general code reasoning)

| Dataset | Author | License |
|---|---|---|
| CodeFeedback-Filtered-Instruction | m-a-p | Apache 2.0 |
| OpenCodeInstruct | NVIDIA | CC BY 4.0 |
| LeetCodeDataset | newfacade | MIT |
| python-codes-25k | flytech | Apache 2.0 |
| CodeAlpaca-20k | sahil2801 | Apache 2.0 |

**Layer 2 — Domain Injection** (embedded systems specialization)

| Dataset | Author | License |
|---|---|---|
| Electrical-engineering | STEM-AI-mtl | MIT |
| stm32-hal-dataset | MuratKomurcu | MIT |
| Hand-curated Arduino/ESP32 | C-28 (original) | Apache 2.0 |
| Temperature-humidity-device | eluri-anilcharles-28 | Apache 2.0 |
| RFID-BASED-SECURITY-SYSTEM | eluri-anilcharles-28 | Apache 2.0 |
| RFID-Reader | eluri-anilcharles-28 | Apache 2.0 |
| arduino-projects | mattiasjahnke | MIT |
| ARDUINO-projects | MadhavBahl | MIT |
| ThatProject | 0015 | Apache 2.0 |
| esp32-mqtt | tuanpmt | Apache 2.0 |
| ESP32-Projects | shameermohamed | custom |

Total datasets added ~37000
---

## System Prompt Architecture

AEGIS uses a runtime-injected system prompt. The prompt is not baked into weights — it is loaded from `system_prompt.md` at inference time. This allows behavioral updates without retraining.

The prompt uses XML-tagged sections:

```
<aegis_identity>     — model identity and role
<aegis_code_principles> — 11 non-negotiable embedded coding rules
<aegis_debugging_protocol> — classify → root cause → mechanism → fix → verify
<aegis_response_format>    — platform → approach → code → notes
<aegis_tone> - to make it direct and concise
<aegis_constraints>
<aegis_identity_responses>
<aegis_easter_eggs> 
```

## Repository Structure

```
AEGIS/
├── README.md
├── LICENSE                          Apache 2.0
├── model_card.md
├── requirements.txt
├── paper/
│   ├── AEGIS_Research_Paper.pdf 
├── dataset/
│   ├── credits.md                   Dataset credits and licenses
│   └── dataset.json                 
├── inference/
│   ├── run_aegis.py                 Basic inference script
│   ├── system_prompt.md             Runtime-injected behavioral config           
│   ├── app.py                       Gradio UI
│   └── requirements.txt
├── training/
│   └── AEGIS_V1_notebook.py         Full training notebook
└── images/
    ├── loss_graph.png
    ├── grad_norm_graph.png
    └── lr_graph.png
```

---

## Known Limitations

- No validation split — training loss only, generalization unverified
- Single epoch (~1.06) — unknown if 0.1637 is true convergence floor
- Trained at 1024 tokens — long-context performance untested
- LeetCode domain bleed — Python algorithm patterns may surface without system prompt
- No formal benchmark evaluation — addressed in V2.0

---

## Citation

```bibtex
@misc{aegis2026,
  title         = {A.E.G.I.S: Domain-Specialized QLoRA Fine-Tuning
                   for Embedded Systems Code Generation},
  author        = {C-28 and A-47},
  year          = {2026},
  month         = {June},
  note          = {Independent Researchers. No institutional affiliation.},
  url           = {https://github.com/eluricharles/AEGIS},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```

---

## Acknowledgements

The authors thank the open-source community whose datasets made this work possible, including m-a-p, NVIDIA, and the individual contributors listed in `dataset/README.md`. Training was tracked using Weights & Biases. Writing assistance was provided by AI language model tools; all scientific content, experimental design, and results are the authors' own. This work was conducted without institutional funding or compute resources.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

*Built on free hardware. No institution. No shortcuts on the parts that matter.*
*— C-28 & A-47*
