# Industrial Inspection Log-to-Narrative Generator (`LING3000`)

A custom PyTorch GPT-style autoregressive transformer that translates structured computer vision hazard-detection logs into clear, human-readable inspection summaries for facility surveyors.

Developed as a computational linguistics research project for **LING 3000** at the University of Connecticut, bridging the NLP reporting layer of a Computer Science Senior Design Capstone sponsored by **Hartford Steam Boiler**.

---

## Overview

Industrial computer vision pipelines output raw, repetitive JSON telemetry during facility video walkthroughs (timestamps, detected boiler components, defect classifications, and confidence scores). While structured JSON is ideal for downstream databases, field surveyors need concise, natural-language narrative summaries to review findings quickly and compile safety reports.

This repository implements an end-to-end conditional language modeling pipeline trained from scratch in PyTorch to convert structured JSON detection events into fluent, domain-accurate inspection prose.

### Example Input & Output

**Input (Structured CV Telemetry):**
```json
{
  "timestamp": "05:22",
  "object": "boiler body",
  "issue": "oil leak",
  "conf": 0.92
}
```

**Output (Generated Surveyor Narrative):**
> *"At 05:22, a boiler body was positively identified with an oil leak (high confidence)."*

---

## Key Features & Architecture

* **Custom PyTorch Transformer:** Decoder-only GPT-style architecture built from the ground up using PyTorch (`torch.nn`), featuring multi-head causal self-attention, learned positional embeddings, layer normalization, and residual feed-forward blocks.
* **Domain-Specific Tokenization & Conditioning:** Formats structured key-value detection logs (`timestamp`, `object`, `issue`, `conf`) into conditioned prompt sequences paired with target narrative descriptions.
* **Confidence-Aware Lexical Mapping:** Translates continuous confidence scores (`conf`) into calibrated linguistic certainty markers (e.g., *"positively identified"*, *"potential"*, *"low-confidence observation"*).
* **Synthetic Corpus Generation:** Includes data generation utilities to pair randomized industrial equipment states, timestamps, and defect classes with varied syntactic templates to prevent mode collapse and repetitive phrasing.

---

## Tech Stack

* **Language:** Python 3.10+
* **Deep Learning:** PyTorch
* **Data Processing:** NumPy, JSON
* **Upstream Integration:** YOLOv8 Object Detection Pipeline (UConn Senior Design Capstone — Hartford Steam Boiler)

---

## Project Structure

```text
ling3000/
├── data/               # Synthetic training pairs & validation JSON logs
├── model/              # PyTorch GPT architecture (attention, transformer blocks)
├── tokenizer/          # Vocabulary builder & JSON-to-token encoding utilities
├── train.py            # Training loop, loss tracking, and checkpoint saving
├── generate.py         # Inference script for translating JSON logs to text
└── README.md
```

---

## Quickstart

### 1. Install Dependencies
```bash
pip install torch numpy
```

### 2. Train the Model
```bash
python train.py
```

### 3. Run Inference on Detection Logs
```bash
python generate.py --input '{"timestamp": "05:22", "object": "boiler body", "issue": "oil leak", "conf": 0.92}'
```

---

## Context & Acknowledgments

* **Course:** LING 3000 — University of Connecticut (Fall 2026)
* **Capstone Integration:** Serves as the natural language generation (NLG) reporting module for the UConn CSE Senior Design **AI Hazard Detection of Industrial Equipment** project (sponsored by **Hartford Steam Boiler**).
```
