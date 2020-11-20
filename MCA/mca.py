from MCA.net_moudle import FC,MLP,LayerNorm
import torch
import torch.nn as nn

import math
import torch.nn.functional as F

class MultiHeadAtt(nn.Module):
    def __init__(self,hidden_size,dropout,num_head,head_hidden_size):
        super(MultiHeadAtt, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.num_head = num_head
        self.head_hidden_size = head_hidden_size
        self.linear_v = nn.Linear(hidden_size,hidden_size)
        self.linear_v = nn.Linear(hidden_size, hidden_size)
        self.linear_v = nn.Linear(hidden_size, hidden_size)
        self.linear_merge = nn.Linear(hidden_size, hidden_size)

    def att(self,v,k,q,mask):
        dim_k = q.size(-1)
        score = torch.matmul(q,k.transpose(-2,1))/torch.sqrt(dim_k)
        if mask is not None:
            score.masked_fill_(mask,-1e9)
        attention_map  = F.softmax(score,dim=-1)
        attention_map = self.dropout(attention_map)
        out = torch.matmul(attention_map,v)
        return  out


    def forward(self,v,k,q,mask):
        batch_size = q.size(0)
        v = self.linear_v(v).view(batch_size,-1,self.num_head,self.head_hidden_size).transpose(1,2)
        q = self.linear_v(q).view(batch_size, -1, self.num_head, self.head_hidden_size).transpose(1,2)
        k = self.linear_v(k).view(batch_size, -1, self.num_head, self.head_hidden_size).transpose(1,2)

        att = self.att(v,k,q,mask)
        att = att.transpose(1,2).contiguous().view(batch_size,-1,self.hidden_size)

        out = self.linear_merge(att)

        return  out

## Feed'Forward Network

class FFNN(nn.Module):
    def __init__(self,hidden_size,ff_size,dropout):
        super(FFNN, self).__init__()
        self.MLP = MLP(hidden_size,ff_size,hidden_size,dropout,use_relu=True)

    def forward(self,x):
        return self.MLP(x)

#  self attention

class SelfAttention(nn.Module):
    def __init__(self,hidden_size,dropout,num_head,head_hidden_size,ff_size):
        super(SelfAttention, self).__init__()
        self.MHAtt = MultiHeadAtt(hidden_size,dropout,num_head,head_hidden_size)
        self.ffnn = FFNN(hidden_size,ff_size,dropout)

        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = LayerNorm(hidden_size)

    def forward(self,x,mask):
        x = self.norm1(x+self.dropout1(self.MHAtt(x,x,x,mask)))
        out = self.norm2(x+self.dropout2(self.ffnn(x)))
        return out


# self Guided attention （X=Y)

class SelfGuidedAttention(nn.Module):
    def  __init__(self,hidden_size,dropout,num_head,head_hidden_size,ff_size):
        super(SelfGuidedAttention, self).__init__()
        self.MHAtt1 = MultiHeadAtt(hidden_size,dropout,num_head,head_hidden_size,ff_size)
        self.MHAtt2 = MultiHeadAtt(hidden_size,dropout,num_head,head_hidden_size,ff_size)
        self.ffnn = FFNN(hidden_size,ff_size,dropout)

        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = LayerNorm(hidden_size)
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = LayerNorm(hidden_size)

    def forward(self,x,y,mask_x,mask_y):
        x = self.norm1(x+self.dropout1(self.MHAtt1(x,x,x,mask_x)))

        x = self.norm2(x+self.dropout2(self.MHAtt2(y,y,x,mask_y)))

        out = self.norm3(x+self.dropout3(self.ffnn(x)))

        return out

# 利用self-attention和self-guided attention 级联形成encoder-decoder结构

class Cascade_MCA(nn.Module):

    def __init__(self,hidden_size,dropout,num_head,head_hidden_size,ff_size,num_layer):
        super(Cascade_MCA, self).__init__()
        self.encoder_list = nn.ModuleList([SelfAttention(hidden_size,dropout,num_head,hidden_size,ff_size) for _ in range(num_layer)])

        self.decoder_list = nn.ModuleList([SelfGuidedAttention(hidden_size, dropout, num_head, hidden_size, ff_size) for _ in range(num_layer)])

    def forward(self,x,y,mask_x,mask_y):
        for enc in self.encoder_list:
            x = enc(x,mask_x)

        for dec in self.decoder_list:
            y = dec(y,x,mask_y,mask_x)

        return x,y