# WiFi Low-Sampling Signal Decoding — Daniel Branch

> “Imagine you have a cheap WiFi receiver — one that only samples signals at 5 MHz instead of the full 20 or 80 MHz used in modern 802.11 standards. Can you still decode packet information accurately enough to be useful?”

This project investigates whether critical WiFi packet information can be recovered from severely undersampled IQ signals.

## Problem Overview

Every WiFi packet begins with a 24-bit header called the L-SIG field. It encodes two critical values:

- Transmission data rate
- Packet length

Accurate decoding of this field is essential. Without it, a receiver cannot correctly interpret or schedule communication.

**Goal:** Decode the 24-bit L-SIG field from low-resolution 5 MHz IQ signals, where much of the original signal information is lost.

## Model & Experiment Structure

### Model 1 — 1D CNN Baseline (`src/`)

- Initial sanity-check model
- Directly operates on raw IQ signals

### Knowledge Distillation — Teacher to Student (`zkd/`)

- Teacher model trained on high-resolution signals
- Student model learns from low-resolution inputs
- Tests whether missing information can be transferred

### Model 2 — DeepMon with ReLU (`src2/`)

- Paper reimplementation of the DeepMon architecture
- Serves as the baseline reproduction

### Model 3 — DeepMon with SiLU (`src3/`)

- Activation upgraded from ReLU to SiLU
- Improves training stability and representation quality
- Serves as the refined baseline

### Model 4 — RIS-MAE (`src3/`)

- Single-modal masked autoencoder
- Learns latent structure from corrupted low-resolution signals
- Focuses on representation learning under information loss

### Model 5 — CrossMAE (`src4/`)

- Cross-modal masked autoencoder
- Main contribution of this branch
- Leverages relationships between different signal views
- Designed to improve robustness under low-sampling conditions

## Additional Experiments (`src4/`)

### Few-Shot Learning

- Evaluates label efficiency
- Measures performance with limited supervised data

### Structured Decode

- Applies domain-knowledge-based post-processing
- Improves prediction consistency without changing the model

## Key Insight

This project explores a core question:

> When signal information is lost due to low sampling rates, is the limitation in the model, or in the data itself?
