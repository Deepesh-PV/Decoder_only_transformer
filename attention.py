import torch
import torch.nn as nn
from torch.nn import Softmax 

class Attention(nn.Module):
    def __init__(self,d_model=2):
        super().__init__()
        self.W_q=nn.Linear(in_features=d_model,out_features=d_model,bias=False,dtype=float)
        self.W_k=nn.Linear(in_features=d_model,out_features=d_model,bias=False,dtype=float)
        self.W_v=nn.Linear(in_features=d_model,out_features=d_model,bias=False,dtype=float)
        self.row_dim=0
        self.col_dim=1
        self.softmax=Softmax(dim=self.col_dim)
    def forward(self,encodings_q:torch.Tensor,encodings_k:torch.Tensor,encodings_v:torch.Tensor,mask=None):
        q=self.W_q(encodings_q)
        k=self.W_k(encodings_k)
        v=self.W_v(encodings_v)

        sims=torch.matmul(q,k.transpose(dim0=self.row_dim,dim1=self.col_dim))
        scaled_sim=sims/torch.tensor(k.shape[self.col_dim]**0.5)
        if mask is not None:
            scaled_sim=scaled_sim.masked_fill(mask=mask,value=-1e9)
        attention_percents=self.softmax(scaled_sim)
        attention_scores=torch.matmul(attention_percents,v)
        return attention_scores

