PAD = 0
UNK = 1

WORD = {
    UNK: '<unk>',
    PAD: '<pad>'
}

# 字典中的占位符

textCNN_param = {
    "filter_size":[2,3,5],
    "embed_dim": 128,
    "filter_num":64
}

biLSTM_param = {
"embed_dim": 64,
"hidden_size" :32,
"bidirectional" : True,

"attention_size" : 16,
"sequence_length" : 16
}
# 通用参数
lr = 0.001
epochs = 1000
batch_size = 64
seed = 1111
cuda_able = True
save = './bilstm_attn_model'
data = './data/corpus.pt'
dropout = 0.5
weight_decay=0.001