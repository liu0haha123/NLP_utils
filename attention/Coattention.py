import torch
import torch.nn as nn
from maskSoftmax import masksoftmax
class AlternativeCoattention(nn.Module):
    def __init__(self,query_input_dim,img_input_dim,emb_dim,query_seq_len,img_seq_len):
        super(AlternativeCoattention, self).__init__()
        self.query_input_dim = query_input_dim
        self.img_input_dim = img_input_dim
        self.emb_dim = emb_dim
        self.query_seq_len= query_input_dim
        self.img_seq_len = img_seq_len


        self.Linear_h1 = nn.Linear(self.query_input_dim,self.emb_dim)
        self.Linear_a1 = nn.Linear(self.emb_dim,1)
        # 第二轮迭代的计算
        self.Linear_h2_1 = nn.Linear(self.img_input_dim,self.emb_dim)
        self.Linear_h2_2 = nn.Linear(self.query_input_dim,self.emb_dim)
        self.Linear_a2 = nn.Linear(self.emb_dim,1)
        # 第三轮迭代
        self.Linear_h3_1 = nn.Linear(self.query_input_dim,self.emb_dim)
        self.Linear_h3_2 = nn.Linear(self.img_input_dim,self.emb_dim)
        self.Linear_a3 = nn.Linear(self.emb_dim,1)

    def forward(self,query_feature,img_feature,mask):
        """

        :param query_feature:
        :param img_feature:
        :param mask:
        :return:
        """
        # 第一个time_step X=V g=0
        # H = tanh(WX*X)
        H1 = self.Linear_h1(query_feature)
        H1 = nn.Dropout(nn.Tanh()(H1))# batch_size,seq_len,emb_dim
        #alpha^x = softmax(W_{hx}^T * H)
        A1 = self.Linear_a1(H1)
        A1 = masksoftmax(A1.view(-1,self.query_seq_len),mask) #A1:(N,seq_len)

        # S1 = \sum alpha^x_i*x_i

        S1 = A1.unsqueeze(1)*query_feature
        S1 = torch.sum(S1,dim=1)  # (N, input_ques_dim)

        # 第二个time step
        # H2 = tanh(X * W_x + (W_g * s1) * 1^T)
        H2_1  = self.Linear_h2_1(img_feature) #batch,img_seq_len,emb_dim
        H2_2 = self.Linear_h2_2(S1)
        H2_2 = H2_2.unsqueeze(1).expand(-1,self.img_seq_len,self.emb_dim)# same as H2_1

        H2 = nn.Dropout(0.5)(nn.Tanh()(H2_1+H2_2))

        # alpha2
        A2 = self.Linear_a2(H2)
        A2 = nn.Softmax(1)(A2.view(-1,self.img_seq_len))

        #V2
        V2 = A2.unsqueeze(-1)*img_feature
        V2 = torch.sum(V2,dim=1)

        # 第三轮以后的迭代

        H3_1 = self.Linear_h3_1(query_feature)
        H3_2 = self.Linear_h3_2(V2)
        H3_2 = H3_2.unsqueeze(1).expand(-1,self.query_seq_len,self.emb_dim)
        H3 = nn.Dropout(0.5)(nn.Tanh()(H3_2+H3_1))

        # alpha3

        A3 = self.Linear_a3(H3)
        A3 = nn.Softmax()(A3.view(-1,self.query_seq_len))

        S3 = A3.unsqueeze(-1)*query_feature
        S3 = torch.sum(S3,dim=1)


        return S3,V2