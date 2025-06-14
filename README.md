# PARA-PHRASER: Fine-Tuned T5 Paraphrasing Model with LoRA

![Demo Screenshot 1](./Output/1.png)
![Demo Screenshot 2](./Output/2.png)

## 🔍 Overview

A lightweight yet powerful paraphrasing tool powered by a fine-tuned `t5-small` model with LoRA (Low-Rank Adaptation) for efficient parameter tuning. Achieves high-quality paraphrasing while running efficiently on CPU.

## ✨ Key Features

- **Efficient Fine-Tuning**: LoRA configuration with rank=16 (r=16)
- **CPU-Friendly**: Optimized to run smoothly on local machines
- **Web Interface**: Clean FastAPI backend with responsive HTML/CSS/JS frontend
- **Quality Results**: Produces fluent, diverse paraphrases while preserving meaning

## 🛠️ Technical Details

### Model Architecture
- Base Model: `t5-small` (60M parameters)
- Fine-Tuning Method: LoRA (Low-Rank Adaptation)
  - Rank: 16
  - Target Modules: Attention q/v layers
- Training Data: Custom paraphrasing dataset

### Performance
- Inference Time (CPU): ~0.5-1 seconds per sentence
- Memory Usage: <1GB RAM
- Output Quality: Comparable to larger models for most use cases
