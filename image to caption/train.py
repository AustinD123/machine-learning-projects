from main_model import fullmodel
from dataload import train_loader, test_loader
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DataParallel
import os
from transformers import BertTokenizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn, optim

tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

model = fullmodel(
    embed_size=300,
    attention_dim=256,
    encoder_dim=2048,
    decoder_dim=512
).to(device)

