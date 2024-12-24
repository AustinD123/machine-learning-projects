# Transformer Components in PyTorch

This repository contains the implementation of key components used in a Transformer architecture, written in PyTorch. The components include input embeddings, positional encoding, and layer normalization. These components are commonly used in models for tasks such as language modeling, machine translation, and more.

## Overview

The code defines three key classes:

1. **`InputEmbeddings`**: Creates an embedding layer that maps the input token indices to dense vectors, scaled by the square root of the model's dimension (`d_model`).
2. **`PositionalEncoding`**: Adds positional information to the token embeddings, making the model aware of the order of tokens in the sequence.
3. **`LayerNormalization`**: Performs layer normalization, which helps stabilize and speed up training.

## Classes

### `InputEmbeddings`

This class implements an embedding layer for the input tokens. The embeddings are scaled by the square root of the model's dimension to prevent the gradients from becoming too small.

#### Constructor Parameters:
- `d_model (int)`: The dimensionality of the output embeddings.
- `vocab_size (int)`: The size of the vocabulary (i.e., the number of unique tokens).

#### Methods:
- `forward(x)`: Given an input tensor `x` (token indices), it returns the scaled embeddings.

### `PositionalEncoding`

This class generates positional encodings, which are added to the input embeddings to inject information about the positions of tokens in a sequence.

#### Constructor Parameters:
- `d_model (int)`: The dimensionality of the input embeddings and positional encodings.
- `seq_len (int)`: The maximum sequence length that the model can handle.
- `dropout (float)`: Dropout rate applied to the final output.

#### Methods:
- `forward(x)`: Adds positional encodings to the input tensor `x` and applies dropout.

### `LayerNormalization`

This class implements layer normalization, which normalizes the input tensor along the last dimension. Layer normalization helps to stabilize the training of deep networks by reducing the internal covariate shift.

#### Constructor Parameters:
- `eps (float)`: A small value added to the denominator to prevent division by zero (default is `1e-6`).

#### Methods:
- `forward(x)`: Normalizes the input tensor `x` and applies the learnable scaling (`alpha`) and bias (`bias`) parameters.

## Requirements

- Python 3.x
- PyTorch 1.7.0 or later
- `math` library (included in Python's standard library)

## Usage Example

Here is an example of how to use these components:

```python
import torch
import math

# Example parameters
d_model = 512
vocab_size = 10000
seq_len = 100
dropout = 0.1

# Create instances of the components
input_embeddings = InputEmbeddings(d_model=d_model, vocab_size=vocab_size)
positional_encoding = PositionalEncoding(d_model=d_model, seq_len=seq_len, dropout=dropout)
layer_norm = LayerNormalization()

# Sample input (batch of sequences with token indices)
x = torch.randint(0, vocab_size, (32, seq_len))  # Example input: batch of sequences of token indices

# Get input embeddings
embedding_output = input_embeddings(x)

# Add positional encoding
positional_output = positional_encoding(embedding_output)

# Apply layer normalization
normalized_output = layer_norm(positional_output)

print(normalized_output.shape)  # Output shape: (32, seq_len, d_model)
