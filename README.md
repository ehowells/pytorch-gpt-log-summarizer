# Industrial Inspection Log-to-Narrative Generator (`LING3000`)

A custom PyTorch GPT-style autoregressive transformer that translates structured computer vision hazard-detection logs into clear, human-readable inspection summaries for facility surveyors.

Developed as a computational linguistics research project for **LING 3000** at the University of Connecticut, bridging the NLP reporting layer of a Computer Science Senior Design Capstone sponsored by **Hartford Steam Boiler**.

---

## Overview

Industrial computer vision pipelines output raw, repetitive JSON telemetry during facility video walkthroughs (timestamps, detected boiler components, defect classifications, and confidence scores). While structured JSON is ideal for downstream databases, field surveyors need concise, natural-language narrative summaries to review findings quickly and compile safety reports.

This repository implements an end-to-end sequence-to-sequence / conditional language modeling pipeline trained from scratch in PyTorch to convert structured JSON detection events into fluent, domain-accurate inspection prose.
