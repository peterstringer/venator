"""Train/validation/test splitting with strict methodology constraints.

Enforces the unsupervised anomaly detection methodology:
- Training set: benign prompts ONLY (detector learns "normal")
- Validation set: benign prompts ONLY (for threshold tuning)
- Test set: held-out benign + ALL jailbreak prompts

Jailbreak prompts must NEVER appear in training or validation data.
"""
