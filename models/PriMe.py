import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder,EncoderLayer,Cross_EncoderLayer
from layers.Embed import Aug_Channel_Embedding,Aug_Temporal_Embedding,init_query

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.c_layer= configs.c_layer
        self.t_layer= configs.t_layer
        self.llm_layer= configs.llm_layer
        self.channel_encoder = nn.Sequential(Aug_Channel_Embedding(configs),Encoder([EncoderLayer(configs) \
                                            for _ in range(self.c_layer)])) if self.c_layer>0 else nn.Identity()
        self.temporal_encoder = nn.Sequential(Aug_Temporal_Embedding(configs),Encoder([EncoderLayer(configs) \
                                            for _ in range(self.t_layer)])) if self.t_layer>0 else nn.Identity()
        
        self.llm_encoder = nn.Sequential(Aug_Channel_Embedding(configs,768),Encoder([EncoderLayer(configs) \
                                            for _ in range(self.llm_layer)])) if self.llm_layer>0 else nn.Identity()
        
        self.Cross_Encoder = Cross_EncoderLayer(configs)
        self.query = init_query(configs.d_model, configs.init)
        self.projector = nn.Linear(configs.d_model,configs.num_class)

    def forward(self, x_enc,embeddings):
        B,T,N= x_enc.shape
        channel= self.channel_encoder(x_enc) if self.c_layer>0 else 0
        temporal= self.temporal_encoder(x_enc) if self.t_layer>0 else 0
        
        llm=self.llm_encoder(embeddings) if self.llm_layer>0 else 0
        
        
        channel=self.Cross_Encoder(self.query.repeat(B, 1).unsqueeze(1),channel,channel) if self.c_layer>0 else 0
        temporal=self.Cross_Encoder(self.query.repeat(B, 1).unsqueeze(1),temporal,temporal) if self.t_layer>0 else 0
        llm=self.Cross_Encoder(self.query.repeat(B, 1).unsqueeze(1),llm,llm) if self.llm_layer>0 else 0
        
        # B N D -> B D -> B C 
        logits = self.projector((channel+temporal+llm).squeeze(1))
        return logits
