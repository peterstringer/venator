"""MLX-based hidden state extraction from transformer models.

Hooks into transformer layer forward passes to capture hidden state activations.
Uses MLX for Apple Silicon-optimized inference with 4-bit quantized models.
"""
