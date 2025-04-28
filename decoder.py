import torch
import lightning as L
import torch.nn as nn
from torch.optim import Adam
from position import positionalembedding
from attention import Attention
class Decoder(nn.Module):
    def __init__(self,num_tokens,d_model,max_len):
        super().__init__()
        self.we=nn.Embedding(num_embeddings=num_tokens,embedding_dim=d_model)
        self.pe=positionalembedding(d_model=d_model,max_len=max_len)
        self.self_attention=Attention(d_model=d_model)
        self.fc_layer=nn.Linear(in_features=d_model,out_features=num_tokens,dtype=float)
        
    def forward(self,token_ids):
        word_embeddings=self.we(token_ids)
        print("word_embedding shape:",word_embeddings.shape)
        position_encoded=self.pe(word_embeddings)
        print("position shape:",position_encoded.shape)
        mask=torch.tril(torch.ones((token_ids.shape[0],token_ids.shape[0])))
        mask=mask==0
        self_attention_value=self.self_attention(position_encoded,position_encoded,position_encoded,mask=mask)
        print("self attention shape:",self_attention_value.shape)
        residual_connection=position_encoded+self_attention_value
        print("residual connextion shape:",residual_connection.shape)
        return self.fc_layer(residual_connection)
    



