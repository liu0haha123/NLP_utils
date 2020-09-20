import torch
import torch.nn.functional as F

def masksoftmax(data,mask):
    data.masked_fill_(mask,-999999)
    return F.softmax(data,1)
