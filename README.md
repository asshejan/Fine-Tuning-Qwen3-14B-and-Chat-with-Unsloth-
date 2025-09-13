# Fine-Tuning-Qwen3-14B-and-Chat-with-Unsloth-
This repository demonstrates how to efficiently fine-tune the Qwen3-14B model for both mathematical reasoning and conversational chat using the Unsloth library. The process involves mixed-dataset preparation, LoRA configuration, training, and saving the model in multiple formats, including GGUF for llama.cpp.

# Fine-Tuning Qwen3-14B for Reasoning and Chat with Unsloth

![Unsloth Logo](https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png)

## 📖 Overview

This project provides a complete walkthrough for fine-tuning the **Qwen3-14B** large language model using the **Unsloth** library. The goal is to create a model proficient in both specialized mathematical reasoning and general-purpose conversation.

The notebook leverages Unsloth's optimizations to enable fast, memory-efficient training on a free Google Colab Tesla T4 GPU. It covers the entire pipeline from data preparation and model training to inference and saving the final model in various formats.

---

## ✨ Key Features

-   **🚀 Accelerated Training:** Utilizes Unsloth for up to 2x faster training and 60% less memory usage.
-   **🧠 Mixed-Dataset Strategy:** Combines a mathematical reasoning dataset (`unsloth/OpenMathReasoning-mini`) with a general conversational dataset (`mlabonne/FineTome-100k`) to build a versatile model.
-   **🛠️ LoRA Fine-Tuning:** Implements Low-Rank Adaptation (LoRA) to efficiently update a small fraction of the model's parameters.
-   **💬 Dual Inference Modes:** Demonstrates how to generate responses in both a high-accuracy "reasoning" mode and a standard "chat" mode.
-   **💾 Multiple Export Options:** Includes code to save the fine-tuned model as LoRA adapters, merged 16-bit/4-bit models for VLLM, and in GGUF format for use with llama.cpp and Ollama.

---

## 🚀 Getting Started

You can run this project directly in Google Colab.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(14B)_Reasoning_Conversational.ipynb)

### Prerequisites

-   A Google account to access Google Colab.
-   A **Tesla T4 GPU** runtime (available for free in Colab).

### ⚙️ Installation

The notebook handles all dependencies. The primary libraries are installed using `pip`:

```python
# Installs Unsloth, Transformers, TRL, PEFT, and other required packages
!pip install "unsloth[colab-new] @ git+[https://github.com/unslothai/unsloth.git](https://github.com/unslothai/unsloth.git)"
!pip install --no-deps transformers==4.55.4 peft==0.12.0 trl==0.10.0 accelerate==0.32.0 bitsandbytes==0.43.1
