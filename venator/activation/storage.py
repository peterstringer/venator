"""HDF5-based storage for activation matrices.

Provides efficient read/write for large activation arrays extracted from LLM
hidden states. Uses h5py for chunked, compressed storage suitable for the
high-dimensional activation data (n_samples x n_layers x hidden_dim).
"""
