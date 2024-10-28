# AlexNet Image Classification with PyTorch

This repository contains an implementation of the AlexNet architecture using PyTorch for image classification. The model uses pretrained weights from ImageNet and can classify images into 1000 different categories.

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture Details](#model-architecture-details)
- [Data Preprocessing](#data-preprocessing)

## Overview

This implementation uses the AlexNet architecture, a pioneering deep convolutional neural network that significantly improved image classification accuracy when it was introduced. The model is pretrained on ImageNet and can classify images into 1000 different categories.

## Architecture

AlexNet consists of 8 layers:
- 5 convolutional layers
- 3 fully connected layers
- ReLU activations
- MaxPooling layers
- Dropout for regularization

```
AlexNet(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=11, stride=4, padding=2)  # Layer 1
    (1): ReLU(inplace=True)
    (2): MaxPool2d(kernel_size=3, stride=2)
    
    (3): Conv2d(64, 192, kernel_size=5, padding=2)  # Layer 2
    (4): ReLU(inplace=True)
    (5): MaxPool2d(kernel_size=3, stride=2)
    
    (6): Conv2d(192, 384, kernel_size=3, padding=1)  # Layer 3
    (7): ReLU(inplace=True)
    
    (8): Conv2d(384, 256, kernel_size=3, padding=1)  # Layer 4
    (9): ReLU(inplace=True)
    
    (10): Conv2d(256, 256, kernel_size=3, padding=1)  # Layer 5
    (11): ReLU(inplace=True)
    (12): MaxPool2d(kernel_size=3, stride=2)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(6, 6))
  (classifier): Sequential(
    (0): Dropout(p=0.5)
    (1): Linear(9216, 4096)  # FC Layer 1
    (2): ReLU(inplace=True)
    (3): Dropout(p=0.5)
    (4): Linear(4096, 4096)  # FC Layer 2
    (5): ReLU(inplace=True)
    (6): Linear(4096, 1000)  # FC Layer 3 (Output)
  )
)
```

## Requirements

```bash
torch
torchvision
pillow
requests
datasets
```

## Installation

```bash
pip install torch torchvision pillow requests datasets
```

## Usage

```python
# Import required libraries
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Define image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load pretrained model
model = models.alexnet(pretrained=True)
model.eval()

# Load and preprocess image
image = Image.open('your_image.jpg')
img_tensor = transform(image).unsqueeze(0)

# Make prediction
with torch.no_grad():
    output = model(img_tensor)
prediction = torch.argmax(output)

# Get class labels
import requests
labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
labels = requests.get(labels_url).text.split('\n')
predicted_class = labels[prediction]
```

## Model Architecture Details

### Convolutional Layers

1. **First Convolutional Layer**
   - Input: 224x224x3 image
   - Kernel: 11x11
   - Stride: 4
   - Output: 64 feature maps
   - Followed by ReLU and MaxPooling

2. **Second Convolutional Layer**
   - Kernel: 5x5
   - Output: 192 feature maps
   - Followed by ReLU and MaxPooling

3. **Third Convolutional Layer**
   - Kernel: 3x3
   - Output: 384 feature maps
   - Followed by ReLU

4. **Fourth Convolutional Layer**
   - Kernel: 3x3
   - Output: 256 feature maps
   - Followed by ReLU

5. **Fifth Convolutional Layer**
   - Kernel: 3x3
   - Output: 256 feature maps
   - Followed by ReLU and MaxPooling

### Fully Connected Layers

1. **First FC Layer**
   - Input: 6x6x256 = 9216 neurons
   - Output: 4096 neurons
   - Dropout: 0.5
   - ReLU activation

2. **Second FC Layer**
   - Input: 4096 neurons
   - Output: 4096 neurons
   - Dropout: 0.5
   - ReLU activation

3. **Output Layer**
   - Input: 4096 neurons
   - Output: 1000 neurons (classes)

## Data Preprocessing

The implementation uses the following preprocessing steps:
1. Resize images to 256x256
2. Center crop to 224x224
3. Convert to tensor
4. Normalize with ImageNet mean and std:
   - Mean: [0.485, 0.456, 0.406]
   - Std: [0.229, 0.224, 0.225]

