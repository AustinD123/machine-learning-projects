import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.d_model=d_model
        self.vocab_size=vocab_size
        self.embedding=nn.Embedding(d_model,vocab_size)

    def forward(self,x):
        return self.embedding(x)*math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    def __init__(self,d_model:int,seq_len:int,dropout:float)->None:
        super().__init__()
        self.d_model=d_model
        self.seq_len=seq_len
        self.dropout=dropout

        pe=torch.zeros(seq_len,d_model)
        position=torch.arrange(0,seq_len,dtyper=torch.float).unsqueeze(1)
        div_term=torch.exp(torch.arrange(0,d_model,2).float()*(-math.log(10000.0)/d_model))

        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)
        pe=pe.unsqueeze(0)

        self.register_buffer('pe',pe)

    def forward(self,x):
        x=x+(self.pe[:,:x.shape[1],:]).requires_grad_(False)
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    def __init__(self, eps:float=10**-6 )->None:
        super().__init__()
        self.eps=eps
        self.alpha=nn.Parameter(torch.ones(1))
        self.bias=nn.Parameter(torch.zeros(1))
    def forward(self,x):
        mean=x.mean(dim=-1,keepdim=True)
        std=x.std(dim=-1,keepdim=True)
        return self.alpha*(x-mean)/(std+self.eps)+self.bias
    