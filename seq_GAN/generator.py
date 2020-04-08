import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np
import pdb
import math
import torch.nn.init as init


class Generator(nn.Module):
    def __init__(self,vocab_size,hidden_size,emb_dim,max_seq_len,gpu=False, oracle_init=False):
        self.vocab_size = vocab_size
        self.hidden_size =hidden_size
        self.max_seq_len = max_seq_len
        self.emb_dim = emb_dim
        self.gpu = gpu
        self.orcale_init = oracle_init

        self.emb = nn.Embedding(vocab_size,emb_dim)
        self.gru = nn.GRU(emb_dim,hidden_size)
        self.out = nn.Linear(hidden_size,vocab_size)

        # initialise oracle network with N(0,1)
        # otherwise variance of initialisation is very small => high NLL for data sampled from the same model
        if oracle_init:
            for p in self.parameters():
                init.normal(p, 0, 1)


    def init_hidden(self,batch_size=1):
        h = autograd.Variable(torch.zeros(1,batch_size,self.hidden_size))
        if self.gpu:
            return h.cuda()
        else:
            return h


    def forward(self,inputs,hidden):
        # input_dim:batch_size
        emb = self.emb(inputs)#（batch_size,emb_dim)
        emb = emb.view(1,-1,self.hidden_size)# 1 x batch_size x embedding_dim
        output,hidden = self.gru(emb,hidden)# output:1 x batch_size x hidden_dim
        out = self.out(output.view(-1,self.hidden_size))# 1xbatch_sizexvocab_size
        out = F.log_softmax(out,dim=1)
        return out,hidden


    def sample(self,num_samples,start_letters=0):
        """
        返回num_sample个max——len的采样序列
        :param num_samples:
        :param start_letters:
        :return:
        """
        samples = torch.zeros(num_samples,self.max_seq_len).type(torch.LongTensor)
        h = self.init_hidden(num_samples)

        input = autograd.Variable(torch.LongTensor([start_letters]*num_samples))
        if self.gpu:
            h = h.cuda()
            input = input.cuda()
            samples = samples.cuda()
        for i in range(self.max_seq_len):
            out,h = self.forward(input,h)
            out = torch.multinomial(torch.exp(out),1)
            samples[:,i] = out.view(-1).data

            input = out.view(-1)
        return samples

    def batchNLLLoss(self,inputs,targets):
        """
        返回序列的NLLLoss

        :param inputs:
        :param targets:
        :return:
        """
        loss_function = nn.NLLLoss()
        batch_size,seq_len = inputs.size()
        # seq_len ,batch_size
        inputs = inputs.permute(1,0)
        targets = targets.permute(1,0)
        h = self.init_hidden(batch_size)
        loss = 0
        for i in range(seq_len):
            out,h = self.forward(inputs[i],h)
            loss +=loss_function(out,targets[i])
        # 返回每一个batch的loss
        return loss

    def batchPGLoss(self,inputs,targets,reward):
        """
        返回强化学习中梯度策略的 pseudo-loss

        :param inputs:（batch_size,seq_len)
        :param targets: 同上
        :param reward: 一个完整的句子执行判别后的回报价值
        :return:
        """
        batch_size, seq_len = inputs.size()
        inputs = inputs.permute(1, 0)          # seq_len x batch_size
        target = targets.permute(1, 0)    # seq_len x batch_size
        h = self.init_hidden(batch_size)

        loss = 0
        for i in range(seq_len):
            out,h = self.forward(inputs[i],h)
            # todo h.detached
            for j in range(batch_size):
                # log(P(y_t|Y_1:Y_{t-1})) * Q
                loss+=-out[j][target[i][j]]*reward[i]

        return  loss/batch_size