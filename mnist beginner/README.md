## Neural Network Architecture

**Layers:**
1. **Input Layer `a[0]`**: 
   - 784 units (corresponding to each pixel in a 28x28 input image)
2. **Hidden Layer `a[1]`**: 
   - 10 units with ReLU activation
3. **Output Layer `a[2]`**: 
   - 10 units with softmax activation (corresponding to the ten digit classes)

## Forward Propagation

1. **Hidden Layer Computation:**
   - `Z[1] = W[1] @ A[0] + b[1]`
   - `A[1] = ReLU(Z[1])`

2. **Output Layer Computation:**
   - `Z[2] = W[2] @ A[1] + b[2]`
   - `A[2] = softmax(Z[2])`

## Backward Propagation

1. **Output Layer Gradients:**
   - `dZ[2] = A[2] - Y`
   - `dW[2] = (1 / m) * dZ[2] @ A[1].T`
   - `dB[2] = (1 / m) * sum(dZ[2], axis=1, keepdims=True)`

2. **Hidden Layer Gradients:**
   - `dZ[1] = (W[2].T @ dZ[2]) * ReLU'(Z[1])`
   - `dW[1] = (1 / m) * dZ[1] @ A[0].T`
   - `dB[1] = (1 / m) * sum(dZ[1], axis=1, keepdims=True)`

## Parameter Updates

1. **Update Weights and Biases:**
   - `W[2] -= α * dW[2]`
   - `b[2] -= α * dB[2]`
   - `W[1] -= α * dW[1]`
   - `b[1] -= α * dB[1]`

## Variables and Shapes

**Forward Propagation:**
- `A[0] = X`: `784 x m`
- `Z[1] ~ A[1]`: `10 x m`
- `W[1]`: `10 x 784` (since `W[1] @ A[0] ~ Z[1]`)
- `b[1]`: `10 x 1`
- `Z[2] ~ A[2]`: `10 x m`
- `W[2]`: `10 x 10` (since `W[2] @ A[1] ~ Z[2]`)
- `b[2]`: `10 x 1`

**Backward Propagation:**
- `dZ[2]`: `10 x m` (same as `A[2]`)
- `dW[2]`: `10 x 10`
- `dB[2]`: `10 x 1`
- `dZ[1]`: `10 x m` (same as `A[1]`)
- `dW[1]`: `10 x 10`
- `dB[1]`: `10 x 1`
