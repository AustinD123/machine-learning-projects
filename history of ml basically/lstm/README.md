![image](https://github.com/user-attachments/assets/aed3d9ea-9ee3-43f8-bf0f-a6c3ee506351)# Custom LSTM Implementation in PyTorch

This repository provides a custom implementation of an LSTM (Long Short-Term Memory) cell using PyTorch. The LSTM is designed to handle sequence data by maintaining memory through cell and hidden states.

## Overview

An LSTM is a type of recurrent neural network (RNN) that improves long-term dependency handling by using gating mechanisms. This implementation includes:
- **Forget Gate**: Determines what information to discard from the previous cell state.
- **Input Gate**: Decides which new information to add to the cell state.
- **Cell State**: Stores long-term memory.
- **Output Gate**: Decides what to output as the hidden state for the current time step.
  ![main](https://miro.medium.com/v2/resize:fit:828/format:webp/0*VXNy36Ay_Rq3m1LE.png)
  
  

### Key States
- **Hidden State (`ht`)**: Represents short-term memory and influences the output.
- **Cell State (`ct`)**: Stores long-term memory and is updated at each step.

## Code Summary

### Forward Pass
- **Forget Gate**: `ft = sigmoid(Wf * input + Uf * ht)`
![forget](https://miro.medium.com/v2/resize:fit:828/format:webp/0*huVOAJFuhX3bSWH5.png)
- **Input Gate**: `it = sigmoid(Wi * input + Ui * ht)`
![input](https://miro.medium.com/v2/resize:fit:828/format:webp/0*lipRMhGsnzGrEhaS.png)
- **Candidate Memory**: `ctt = tanh(Wc * input + Uc * ht)`
- **Cell State Update**: `ct = ft * ct + it * ctt`
- **Output Gate**: `ot = sigmoid(Wo * input + Uo * ht)`
  ![output](https://miro.medium.com/v2/resize:fit:828/format:webp/0*lirs3HLo70pNlj0O.png)
- **Hidden State Update**: `ht = ot * tanh(ct)`

### Initialization
Hidden and cell states can be initialized to zeros using the `initHidden` method.

## Example Code
```python
import torch

# Initialize LSTM
input_size = 10
hidden_size = 20
lstm = LSTM(input_size, hidden_size)

# Input tensors
input = torch.randn(5, input_size)  # Batch size = 5
ct = torch.zeros(5, hidden_size)
ht = torch.zeros(5, hidden_size)

# Forward pass
ht, ct = lstm(input, ct, ht)

# Initialize states to zero
ct_zero, ht_zero = lstm.initHidden(ct, ht)
