3Dataset

The A.E.G.I.S training dataset is a heterogeneous corpus of approximately 37,000+ samples designed for embedded systems code generation and hardware-aware reasoning.

The dataset combines general code reasoning datasets with embedded-specific sources covering Arduino, ESP32, STM32, AVR Assembly, sensor integration, firmware development, electronics, and debugging workflows.

# Data Sources

- CodeFeedback-Filtered-Instruction
- OpenCodeInstruct (NVIDIA)
- LeetCodeDataset
- python-codes-25k
- CodeAlpaca-20k
- Electrical Engineering Dataset
- STM32 HAL Dataset
- Hand-curated Arduino and ESP32 examples
- Real embedded systems project repositories

# Purpose

The dataset was created to improve performance on:

- Embedded firmware development
- Microcontroller programming
- Hardware-aware code generation
- Peripheral configuration
- Embedded debugging
- Electronics-related reasoning

# Format

Samples are formatted using the Alpaca instruction template:

Instruction → Input → Response

# Built for A.E.G.I.S V1.0 by C-28 & A-47.
