import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import const
import numpy as np

class BiLSTM_Attn(nn.Module):
    def __init__(self,vocab_size,emb_size,lstm_hidden_size,output_size,batch_size,bidirectional, dropout, use_cuda, attention_size, sequence_length):
        """
        
        :param vocab_size:输入输出的词典 
        :param emb_size: 
        :param lstm_hidden_size: 
        :param output_size: 
        :param batch_size: 
        :param bidirectional: 是否使用双向LSTM
        :param dropout: 
        :param use_cuda: 
        :param attention_size: 
        :param sequence_length: 
        """
        super(BiLSTM_Attn, self).__init__()
        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = lstm_hidden_size
        self.vocab_size = vocab_size
        self.embed_dim = emb_size
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.use_cuda = use_cuda
        self.sequence_length = sequence_length

        self.emb = nn.Embedding(self.vocab_size,self.embed_dim)
        self.emb.weight.data.uniform_(-1.0,1.0)

        self.layer_size = 1

        self.LSTM = nn.LSTM(self.embed_dim,self.hidden_size,self.layer_size,dropout=self.dropout,
                            bidirectional=self.bidirectional)

        if self.bidirectional:
            self.layer_size = self.layer_size*2
        else:
            self.layer_size = self.layer_size

        self.attention_size = attention_size

        if self.use_cuda:
            self.w_omega = Variable(torch.zeros(self.hidden_size * self.layer_size, self.attention_size).cuda())
            self.u_omega = Variable(torch.zeros(self.attention_size).cuda())
        else:
            self.w_omega = Variable(torch.zeros(self.hidden_size * self.layer_size, self.attention_size))
            self.u_omega = Variable(torch.zeros(self.attention_size))

        self.output = nn.Linear(lstm_hidden_size * self.layer_size, output_size)


    def attention_net(self,lstm_output):
        # (squence_length, batch_size, hidden_size*layer_size)
        lstm_output_reshape = torch.Tensor.reshape(lstm_output, shape=[-1, self.hidden_size * self.layer_size])
        # (squence_length*batch_size, hidden_size*layer_size)
        # a起一个tanh加线性层的作用
        att_tanh = torch.tanh(torch.mm(lstm_output_reshape,self.w_omega))
        #(seq_len*batch_size,attention_size)
        attn_hidden_layer = torch.mm(att_tanh,torch.Tensor.reshape(self.u_omega, [-1, 1]))
        # print(attn_hidden_layer.size()) = (squence_length * batch_size, 1)
        # 一维的注意力汇聚
        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer), [-1, self.sequence_length])
        #print(exps.size()) = (batch_size, squence_length)
        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
        #print(alphas.size()) = (batch_size, squence_length)
        alphas_reshape = torch.Tensor.reshape(alphas, [-1, self.sequence_length, 1])
        #print(alphas_reshape.size()) = (batch_size, squence_length, 1)
        #print(attn_hidden_layer.size()) = (squence_length * batch_size, 1)
        state = lstm_output.permute(1, 0, 2)
        #print(state.size()) = (batch_size, squence_length, hidden_size*layer_size)

        attn_output = torch.sum(state * alphas_reshape, 1)
        #print(attn_output.size()) = (batch_size, hidden_size*layer_size)

        return attn_output

    def forward(self,inputs,batch_size = None):
        inputs = inputs.long()
        inputs = self.emb(inputs)
        # 因为att——net中需要batch_first
        inputs = inputs.permute(1,0,2)
        if self.use_cuda:
            h_0 = Variable(torch.zeros(self.layer_size, self.batch_size, self.hidden_size).cuda())
            c_0 = Variable(torch.zeros(self.layer_size, self.batch_size, self.hidden_size).cuda())
        else:
            h_0 = Variable(torch.zeros(self.layer_size, self.batch_size, self.hidden_size))
            c_0 = Variable(torch.zeros(self.layer_size, self.batch_size, self.hidden_size))

        lstm_output, (final_hidden_state, final_cell_state) = self.LSTM(inputs, (h_0, c_0))
        attn_output = self.attention_net(lstm_output)
        logits = self.output(attn_output)
        return logits
