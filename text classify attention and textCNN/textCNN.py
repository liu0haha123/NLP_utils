import torch
import torch.nn as nn
import torch.nn.functional as F


import numpy as np


class TextCNN(nn.Module):
    def __init__(self,vocab_size,emb_dim,filter_sizes,num_filter,num_classes,embedding_pretrained,dropout):
        super(TextCNN, self).__init__()

        if embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

        self.convs = nn.ModuleList([nn.Conv2d(1,num_filter,(filter_size,emb_dim)) for  filter_size in filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filter * len(filter_sizes),num_classes)

    def conv_and_pool(self,inputs,conv):
        inputs = F.relu(conv(inputs)).squeeze(3)

        outputs = F.max_pool1d(inputs,inputs.size(2)).squeeze(2)

        return outputs

    def forward(self,inputs):
        embed = self.embedding(inputs)
        out = embed.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
