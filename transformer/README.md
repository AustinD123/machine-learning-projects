# PyTorch Transformer Implementation

## Table of Contents

* [Overview](#overview)
* [Components](#components)
* [Usage](#usage)
* [Implementation Details](#implementation-details)
* [Requirements](#requirements)

## Overview

This repository contains a PyTorch implementation of the Transformer model as described in the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).

## Components

### Core Modules

#### Transformer
```python
class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
```
The main model class combining encoder, decoder, embeddings, and projection layers.

#### Encoder and Decoder
```python
class Encoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList)

class Decoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList)
```
Process input sequences and generate output sequences respectively.

#### Multi-Head Attention
```python
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float)
```
Implements the multi-head attention mechanism supporting both self-attention and cross-attention.

### Supporting Modules

#### Layer Normalization
```python
class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float=10**-6)
```
Normalizes the outputs of sub-layers.

#### Feed Forward Block
```python
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float)
```
Implements position-wise feed-forward networks.

#### Input Embeddings
```python
class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int)
```
Converts input tokens to embeddings.

#### Positional Encoding
```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float)
```
Adds positional information to the embeddings.

## Usage

### Model Creation

```python
transformer = build_transformer(
    src_vocab_size=1000,    # Source vocabulary size
    tgt_vocab_size=1000,    # Target vocabulary size
    src_seq_len=128,        # Max source sequence length
    tgt_seq_len=128,        # Max target sequence length
    d_model=512,            # Model dimension
    N=6,                    # Number of encoder/decoder blocks
    h=8,                    # Number of attention heads
    dropout=0.1,           # Dropout rate
    d_ff=2048              # Feed-forward dimension
)
```

### Forward Pass

```python
# Encoding
encoder_output = transformer.encode(src, src_mask)

# Decoding
decoder_output = transformer.decode(encoder_output, src_mask, tgt, tgt_mask)

# Final projection
output = transformer.project(decoder_output)
```

## Implementation Details

### Architecture Specifications

* **Encoder Stack**: N=6 identical layers
* **Decoder Stack**: N=6 identical layers
* **Attention Heads**: h=8 parallel attention heads
* **Model Dimension**: d_model=512
* **Feed-Forward Size**: d_ff=2048
* **Dropout Rate**: 0.1

### Key Features

* Layer normalization before each sub-layer
* Residual connections around each sub-layer
* Positional encoding using sine and cosine functions
* Scaled dot-product attention
* Xavier uniform initialization

### Attention Computation

The attention mechanism follows these steps:

1. Linear projections of queries, keys, and values
2. Split into multiple heads
3. Scaled dot-product attention
4. Concatenation of heads
5. Final linear projection

### Positional Encoding

Uses sinusoidal position encoding:

* Even indices: sine function
* Odd indices: cosine function
* Wavelengths form geometric progression from 2π to 10000·2π

## Requirements

* Python 3.6+
* PyTorch
* Math (standard library)

## Model Parameters

Default hyperparameters (as per the original paper):

| Parameter | Value | Description |
|-----------|--------|-------------|
| d_model | 512 | Embedding dimension |
| N | 6 | Number of blocks |
| h | 8 | Attention heads |
| d_ff | 2048 | Feed-forward dimension |
| dropout | 0.1 | Dropout rate |