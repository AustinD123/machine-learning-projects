import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import BertTokenizer


class EncoderCNN(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        
        # Freeze all ResNet parameters
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        # Remove the final two layers (pooling and FC)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

    def forward(self, images):
        # Extract features from the ResNet
        features = self.resnet(images)                                    # (batch_size, 2048, 7, 7)
        features = features.permute(0, 2, 3, 1)                          # (batch_size, 7, 7, 2048)
        features = features.view(features.size(0), -1, features.size(-1)) # (batch_size, 49, 2048)
        return features

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        
        # Project both encoder and decoder features to the same space
        self.encoder_proj = nn.Linear(encoder_dim, attention_dim)
        self.decoder_proj = nn.Linear(decoder_dim, attention_dim)
        self.attention_proj = nn.Linear(attention_dim, 1)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        # encoder_out shape: (batch_size, num_pixels, encoder_dim)
        # decoder_hidden shape: (batch_size, decoder_dim)
        
        # Project encoder features
        encoder_projected = self.encoder_proj(encoder_out)  # (batch_size, num_pixels, attention_dim)
        
        # Project decoder hidden state
        decoder_hidden_projected = self.decoder_proj(decoder_hidden).unsqueeze(1)  # (batch_size, 1, attention_dim)
        
        # Calculate attention scores
        attention = self.relu(encoder_projected + decoder_hidden_projected)  # (batch_size, num_pixels, attention_dim)
        attention = self.attention_proj(attention).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(attention)  # (batch_size, num_pixels)
        
        # Weight encoder outputs with attention scores
        weighted_encoder_out = (encoder_out * alpha.unsqueeze(2))  # (batch_size, num_pixels, encoder_dim)
        attention_weighted_encoding = weighted_encoder_out.sum(dim=1)  # (batch_size, encoder_dim)
        
        return attention_weighted_encoding, alpha

class Decoder(nn.Module):
    def __init__(self, encoder_dim,decoder_dim,embed_size,attention_dim,drop):
        self.attention=Attention(encoder_dim,decoder_dim,attention_dim)
        self.embedding=nn.Embedding(len(BertTokenizer.from_pretrained('bert-base-uncased')), embed_size)
        self.init_h=nn.Linear(encoder_dim,decoder_dim)
        self.init_c=nn.Linear(encoder_dim,decoder_dim)
        self.lstm_cell = nn.LSTMCell(embed_size + encoder_dim, decoder_dim, bias=True)
        self.fcn = nn.Linear(decoder_dim, self.embedding.num_embeddings)

    def inithidden(self,encoder_out):
        mean_enc=encoder_out.mean(dim=1)
        h=self.init_h(mean_enc)
        c=self.init_c(mean_enc)
        return h,c

    def forward(self,features,captions):
        embeds=self.embedding(captions)
        h, c=self.inithidden(features)
        