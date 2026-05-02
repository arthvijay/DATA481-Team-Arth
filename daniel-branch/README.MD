📡 WiFi Low-Sampling Signal Decoding — Daniel Branch

“Imagine you have a cheap Wi-Fi receiver — one that only samples signals at 5 MHz instead of the full 20 or 80 MHz used in modern 802.11 standards. Can you still decode packet information accurately enough to be useful?”

This project investigates whether critical WiFi packet information can be recovered from severely undersampled IQ signals.

🧠 Problem Overview

Every WiFi packet begins with a 24-bit header (L-SIG field) that encodes:

Transmission data rate
Packet length

Accurate decoding of this field is essential — without it, a receiver cannot correctly interpret or schedule communication.

👉 Goal:
Decode the 24-bit L-SIG field from low-resolution (5 MHz) IQ signals, where much of the original signal information is lost.

🏗️ Model & Experiment Structure
🔹 Model 1 — 1D CNN Baseline (src/)
Initial sanity-check model
Directly operates on raw IQ signals
🔹 Knowledge Distillation (zkd/)
Teacher model trained on high-resolution signals
Student model learns from low-resolution inputs
Tests whether missing information can be transferred
🔹 Model 2 — DeepMon (ReLU) (src2/)
Reimplementation of DeepMon architecture
Serves as paper baseline reproduction
🔹 Model 3 — DeepMon (SiLU) (src3/)
Activation upgraded from ReLU → SiLU
More stable training and improved representation
Acts as refined baseline
🔹 Model 4 — RIS-MAE (src3/)
Single-modal masked autoencoder
Learns latent structure from corrupted signal
Focus: representation learning under information loss
🔹 Model 5 — CrossMAE (src4/) ⭐ Main Contribution
Cross-modal masked autoencoder
Leverages relationships between different signal views
Designed to improve robustness under low sampling conditions
🧪 Additional Experiments (src4/)
▪️ Few-Shot Learning
Evaluates label efficiency
Measures performance with limited supervision
▪️ Structured Decode
Post-processing using domain knowledge
Improves prediction consistency without changing the model
