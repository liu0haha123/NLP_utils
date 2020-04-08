import torch
import torch.autograd as autograd
import torch.nn as nn
import pdb

# 对输入的序列做真实和生成的二分类

class discriminator(nn.Module):
    def __init__(self,emb_dim,hidden_dim,vocab_size,max_seq_len,gpu=False,dropout=0.3):
        super(discriminator, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.gpu = gpu

        self.emb = nn.Embedding(vocab_size,self.emb_dim)
        self.gru = nn.GRU(emb_dim,hidden_dim,bidirectional=True,num_layers=2,dropout=dropout)
        self.gru2hidden = nn.Linear(2*2*hidden_dim,hidden_dim)
        self.dropout = nn.Dropout(dropout)
        # 二分类
        self.hidden2out = nn.Linear(hidden_dim,1)


    def init_hidden(self, batch_size):
        h = autograd.Variable(torch.zeros(2*2*1, batch_size, self.hidden_dim))

        if self.gpu:
            return h.cuda()
        else:
            return h
    def forward(self,inputs,hidden):
        # inputs:(batch_size,seq_len)
        emb = self.emb(inputs).permute(1,0,2)#(batch_size,seq_len,emb_dim)
        _,hidden = self.gru(emb,hidden)
        hidden = hidden.permute(1,0,2).contiguous() # batch_size x 4 x hidden_dim
        hidden = hidden.view(-1,4*self.hidden_dim)
        out = self.gru2hidden(hidden)#batch_size,4*hidden_dim
        out = torch.tanh(out)
        out = self.dropout(out)
        out = self.hidden2out(out)                                 # batch_size x 1
        out = torch.sigmoid(out)
        return out

    def batchClassify(self,inputs):
        """
        Classifies a batch of sequences.
        输入一个词序列
        输出一个二分类结果
            - inputs: batch_size x seq_len
            - target: batch_size (binary 1/0)
        """

        h = self.init_hidden(inputs.size()[0])
        out = self.forward(inputs, h)
        return out.view(-1)


    def batchBCELoss(self,inputs,targets):
        """
        二分类的二元交叉熵
                    - inp: batch_size x seq_len
            - target: batch_size (binary 1/0)
        :param inputs:
        :param targets:
        :return:
        """
        loss_fn = nn.BCELoss()
        h = self.init_hidden(inputs.size()[0])
        out = self.forward(inputs, h)
        return loss_fn(out, targets)