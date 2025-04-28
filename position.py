import torch
import torch.nn as nn

class positionalembedding(nn.Module):
    def __init__(self,d_model=2,max_len=6):
        super().__init__()
        pe=torch.zeros(size=(max_len,d_model),dtype=float)
        position=torch.arange(start=0,end=max_len,step=1,dtype=float).unsqueeze(1)
        embedding_index=torch.arange(start=0,end=d_model,step=2,dtype=float)
        div_term=1/torch.tensor(1000.0)**((embedding_index)/d_model)
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)
        self.register_buffer('pe',pe)
    def forward(self,word_embeddings):
        return word_embeddings+self.pe[:word_embeddings.shape[0],:]