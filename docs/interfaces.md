# Module Interface Contract

This document defines the interfaces between the major modules of the
Sarcasm-Aware Transformer Fusion system.

The interfaces are based on the project methodology. Where the methodology
does not specify an exact implementation detail, the value is marked
"TBD" and must be agreed before implementation.

---

# 1. End-to-End Pipeline

```text
Raw Data
   ↓
Dataset Loading
   ↓
OCR + Text Processing
   ↓
Image / Text Feature Extraction
   ↓
Sarcasm Module
   ↓
Projection & Alignment
   ↓
Sarcasm-Aware Cross-Modal Fusion
   ↓
Classification Heads
   ↓
Predictions + Explainability
