---
name: lora-finetuning
description: "Parameter-Efficient Fine-Tuning (PEFT/LoRA) design: rank selection, target modules, quantization budget, and adapter evaluation."
category: agent-ml-skills
author: param087
crisp_dm_phase: "Phase 5: Evaluation"
version: 1.0.0
---

# Lora Finetuning

## Overview
Parameter-Efficient Fine-Tuning (PEFT/LoRA) design: rank selection, target modules, quantization budget, and adapter evaluation.

### Target CRISP-DM Phase
**Phase 5: Evaluation**

## Analytical Workflow
1. **Input Validation**: Verify dataset inputs, data types, and required column schema.
2. **Execution**: Apply production-grade best practices, avoiding data leakage and common ML pitfalls.
3. **Evaluation**: Validate output against statistical sanity bounds and deterministic test oracles.
4. **Handoff**: Package artifacts (metrics, tables, models, visualizations) for downstream pipeline stages.

## Code Patterns & Best Practices
- Strict separation between training and test distributions (fit only on train).
- Use vectorized operations in place of row iterations.
- Explicitly log and track assumptions, hyperparameters, and cross-validation variance.

## Common Pitfalls to Avoid
- Leaking preprocessing statistics across cross-validation folds.
- Evaluating imbalanced classification using naive accuracy instead of PR-AUC / F1 / ROC-AUC.
- Neglecting calibration and decision threshold tuning.
