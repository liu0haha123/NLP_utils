import numpy as np
import torch
import torch.nn as nn

class ScaleDotProductAttention(nn.Module):
    def __init__(self,d_k):
        super(ScaleDotProductAttention, self).__init__()
        self.d_k= d_k

    def forward(self,q,k,v,attn_mask):
        """

        :param q:(batch_size,h_head,q_len,d_k)
        :param k:(batch_size,h_head,k_len,d_k)
        :param v:(batch_size,h_head,v_len,d_v)
        :param attn_mask:(batch_size,h_head,q_len(seq_len),k_len(seq_len))

        :return:
        """
        att_score = torch.matmul(q,k.tranpose(-1,-2))/np.sqrt(self.d_k)
        att_score.masked_fill_(attn_mask,1e-9)

        # att_score:(batch_size,n_head,q_len,k_len)

        att_weights = nn.Softmax(att_score,dim=-1)(att_score)
        # att_weightd:(batch_size,n_head,q_len,k_len)

        output = torch.matmul(att_weights,v)
        #output (batch_size,n_head,q_len,d_v)
        return output,att_weights

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,n_heads):
        super(MultiHeadAttention, self).__init__()

        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model//n_heads
        self.d_v = d_model//n_heads

        self.WQ = nn.Linear(d_model,d_model)
        self.WK = nn.Linear(d_model,d_model)
        self.WV = nn.Linear(d_model,d_model)

        self.scaled_dot = ScaleDotProductAttention(self.d_k)

        self.linear = nn.Linear(n_heads*self.d_v,d_model)

    def foward(self,Q,K,V,attn_masks):
        """

        :param Q: (batch_size,q_len,d_model)
        :param K: (batch_size,k_len,d_model)
        :param V: (batch_size,v_len,d_model)
        :param attn_masks:（batch_size,seq_len(q_len),seq_len(k_len))

        :return:
        """

        batch_size = Q.size(0)
        q_heads = self.WQ(Q).view(batch_size,-1,self.n_heads,self.d_k)
        k_heads = self.WK(K).view(batch_size,-1,self.n_heads,self.d_k)
        v_heads = self.WV(V).view(batch_size,-1,self.n_heads,self.d_v)

        # |q_heads| : (batch_size, n_heads, q_len, d_k), |k_heads| : (batch_size, n_heads, k_len, d_k), |v_heads| : (batch_size, n_heads, v_len, d_v)\

        attn_masks = attn_masks.unsqueeze(1).repeat(1,self.n_heads,1,1)
        #attn_masks = (batch_size,n_heads,seq_len,seq_len)

        attn,attn_weights = self.scaled_dot(q_heads,k_heads,v_heads,attn_masks)
        # |attn| : (batch_size, n_heads, q_len, d_v)
        # |attn_weights| : (batch_size, n_heads, q_len, k_len)

        attn = attn.transpose(1,2).contiguos().view(batch_size,-1,self.n_heads*self.d_v)
        #attn(batch_sizw,q_len,n_heads*d_v)
        output = self.linear(attn)
        #output(batch_size,q_len,d_model)

        return output

class PositionWiseFeedForwardNetwork(nn.Module):
    def __init__(self,d_model,d_ff):
        super(PositionWiseFeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(d_model,d_ff)
        self.linear2 = nn.Linear(d_ff,d_model)
        self.relu = nn.ReLU()
    def foreard(self,inputs):
        # |inputs| : (batch_size, seq_len, d_model)

        output = self.relu(self.linear1(inputs))
        # |output| : (batch_size, seq_len, d_ff)
        output = self.linear2(output)
        # |output| : (batch_size, seq_len, d_model)

        return output



class EncoderLayer(nn.Module):
    """
    transformer Encoder 的子层
    """
    def __init__(self,d_model,n_heads,dropout,d_ff):
        """

        :param d_model: 输入张量的维度
        :param n_heads: 注意力的头数
        :param dropout:
        :param d_ff: 前馈层FeedForwardd维度
        """
        super(EncoderLayer, self).__init__()
        self.MHAtt = MultiHeadAttention(d_model,n_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.layernorm1 = nn.LayerNorm(d_model,eps=1e-6)

        self.FFNN = PositionWiseFeedForwardNetwork(d_model,d_ff)

        self.dropout2 = nn.Dropout(dropout)
        self.layernorm2 = nn.LayerNorm(d_model,eps=1e-6)

    def forward(self,inputs,attn_mask):
        # |inputs| : (batch_size, seq_len, d_model)
        # |attn_mask| : (batch_size, seq_len, seq_len)
        # self_attention中Q，K，V都是 Input
        attn,attn_weights = self.MHAtt(inputs,inputs,inputs,attn_mask)
        attn_output = self.dropout1(attn)
        attn_output = self.layernorm1(inputs+attn_output)
        # |attn_outputs| : (batch_size, seq_len(=q_len), d_model)
        # |attn_weights| : (batch_size, n_heads, q_len, k_len)
        ffnn_outputs = self.FFNN(attn_output)
        ffnn_outputs = self.dropout2(ffnn_outputs)
        ffnn_outputs = self.dropout2(attn_output+ffnn_outputs)

        # |ffn_outputs| : (batch_size, seq_len, d_model)

        return ffnn_outputs, attn_weights
class TransformerENcoder(nn.Module):


        #TransformerEncoder is a stack of N(6) encoder layers.

        #Args:
        #    vocab_size (int)    : vocabulary size (vocabulary: collection mapping token to numerical identifiers)
        #    seq_len    (int)    : input sequence length
        #    d_model    (int)    : number of expected features in the input
        #    n_layers   (int)    : number of sub-encoder-layers in the encoder
        #    n_heads    (int)    : number of heads in the multiheadattention models
        #    p_dropout     (float)  : dropout value
        #   d_ff       (int)    : dimension of the feedforward network model
        #    pad_id     (int)    : pad token id

    def __init__(self,vocab_size,seq_len,d_model = 512,n_layers = 6,n_heads = 8,p_dropout = 0.1,d_ff=2048,pad_id = 0):
        """

        :param vocab_size:
        :param seq_len:
        :param d_model:
        :param n_layers:
        :param n_heads:
        :param p_dropout:
        :param d_ff:
        :param pad_id:
        """
        super(TransformerENcoder, self).__init__()
        self.pad_id = pad_id
        self.sinusoid_table = self.get_sinusoid_table(seq_len+1,d_model)
        # sub_layers
        self.emb = nn.Embedding(vocab_size,d_model)
        self.pos_emb = nn.Embedding.from_pretrained(self.sinusoid_table,freeze=True)
        self.sub_layers = nn.ModuleList([EncoderLayer(d_model,n_heads,p_dropout,d_ff) for _ in range(n_layers)])

        # 分类层
        self.Linear = nn.Linear(d_model,2)
        self.Softmax = nn.Softmax(dim=-1)


    def forward(self,inputs):
        # inputs:(batch_size,seq_len)
        position = torch.arange(inputs.size(1),device=inputs.device,dtype=inputs.dtype).repeat(inputs.size(0),1)+1
        position_pad_mask = inputs.eq(self.pad_id)
        position.masked_fill_(position_pad_mask,0)

        outputs = self.emb(inputs)+self.pos_emb(position)
        # outputs:(batch_size,seq_len,d_model)
        attn_pad_mask = self.get_attention_padding_mask(inputs,inputs,self.pad_id)
        attn_weights_list  =[]
        for layer in self.sub_layers:
            outputs,attn_weights = layer(inputs,attn_pad_mask)
            # |outputs| : (batch_size, seq_len, d_model)
            # |attn_weights| : (batch_size, n_heads, seq_len, seq_len)
            attn_weights_list.append(attn_weights)
        outputs,_ = torch.max(outputs,dim=1)
        outputs = self.Softmax(self.Linear(outputs))

        return outputs,attn_weights_list



    def get_sinusoid_table(self,seq_len,d_model):
        # 用三角函数编码序列的相对位置
        def get_angel(pos,i,d_model):
            return pos/np.power(10000,(2*(i//2))/d_model)

        sinusoid_table = np.zeros((seq_len,d_model))
        for pos in range(seq_len):
            for i in range(d_model):
                if i%2==0:
                    sinusoid_table[pos,i] =np.sin(get_angel(pos,i,d_model))
                else:
                    sinusoid_table[pos,i] = np.cos(get_angel(pos,id,d_model))

        return torch.FloatTensor(sinusoid_table)


    def get_attention_padding_mask(self, q, k, pad_id):
        attn_pad_mask = k.eq(pad_id).unsqueeze(1).repeat(1, q.size(1), 1)
        # |attn_pad_mask| : (batch_size, q_len, k_len)

        return attn_pad_mask


