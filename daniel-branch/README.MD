HI THIS IS DANIEL BRANCH

"Imagine you have a cheap Wi-Fi receiver — one that only samples the signal at 5 MHz instead of the full 20
or 80 MHz, which modern 802.11 standards use. Can you still decode the packet header accurately enough to
be useful? That is the core question of this project."
"Every Wi-Fi packet begins with a 24-bit header called the L-SIG field. It tells the receiver two critical things:
how fast the data is being transmitted, and how long the packet is. If you get these wrong, you cannot
correctly receive or schedule anything. Our goal is to decode this 24-bit field from a severely undersampled
IQ signal captured at only 5 MHz."

MODEL 1 — 1D CNN Baseline (src/)
/KNOWLEDGE DISTILLATION — Teacher → Student (zkd/)
/MODEL 2 — DeepMon with ReLU (src2/) — Paper Reimplementation
/MODEL 3 — DeepMon with SiLU (src3/) — The Refined Baseline
/MODEL 4 — RIS-MAE (src3/) — Single-Modal Masked Autoencoder
/MODEL 5 — CrossMAE (src4/) — Cross-Modal MAE — Main Contribution
/FEW-SHOT EXPERIMENTS (src4/) — Label Efficiency
/STRUCTURED DECODE (src4/) — Free Domain Knowledge Post-Processing
