import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm
import numpy as np
class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __len__(self):
        return len(self.word2idx)


class Corpus(object):
    def __init__(self):
        self.dictionary = Dictionary()

    def get_data(self, path, batch_size=20):
        # 添加词到字典
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

                    # 对文件做Tokenize
        ids = torch.LongTensor(tokens)
        token = 0
        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
        num_batches = ids.size(0) // batch_size
        ids = ids[:num_batches * batch_size]
        return ids.view(batch_size, -1)

device = "cuda" if torch.cuda.is_available() else "cpu"

embed_size = 128    # 词嵌入的维度
hidden_size = 1024  # LSTM的hidden size
num_layers = 1
num_epochs = 5      # 迭代轮次
num_samples = 1000  # 测试语言模型生成句子时的样本数
batch_size = 20     # 一批样本的数量
seq_length = 30     # 序列长度
learning_rate = 0.002 # 学习率

corpus = Corpus()
ids = corpus.get_data('train.txt', batch_size)
vocab_size = len(corpus.dictionary)
num_batches = ids.size(1) // seq_length
print(vocab_size)

class RNNLM(nn.Module):
    def __init__(self,vocab_size,embed_size,hidden_size,num_layers):
        super(RNNLM, self).__init__()
        self.emb = nn.Embedding(vocab_size,embed_size)
        self.RNN = nn.LSTM(embed_size,hidden_size,num_layers,batch_first=True)
        self.LN = nn.Linear(hidden_size,vocab_size)

    def forward(self,x,h):
        embed = self.emb(x)

        out ,(h,c)= self.RNN(embed,h)

        out = out.reshape(out.size(0)*out.size(1),out.size(2))

        out = self.LN(out)

        return out,(h,c)
model = RNNLM(vocab_size,embed_size,hidden_size,num_layers).to(device)
# 损失构建与优化
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 反向传播过程“截断”(不复制gradient)
def detach(states):
    return [state.detach() for state in states]



for epoch in range(num_epochs):
    h = torch.zeros(num_layers, batch_size, hidden_size).to(device)
    c = torch.zeros(num_layers, batch_size, hidden_size).to(device)
    states= (h,c)
    for i in range(0,ids.size(1) - seq_length, seq_length):
        inputs = ids[:,i:i+seq_length].to(device)
        targets = ids[:,(i+1):(i+1+seq_length)].to(device)

        states = detach(states)

        outputs,states = model(inputs,states)
        loss = criterion(outputs,targets.reshape(-1))

        model.zero_grad()
        loss.backward()
        clip_grad_norm(model.parameters(),0.5)
        optimizer.step()
        step = (i+1)//seq_length # YOUR CODE HERE
        if step % 100 == 0:
            print('全量数据迭代轮次 [{}/{}], Step数[{}/{}], 损失Loss: {:.4f}, 困惑度/Perplexity: {:5.2f}'
                  .format(epoch + 1, num_epochs, step, num_batches, loss.item(), np.exp(loss.item())))

# 测试语言模型
with torch.no_grad():
    with open('sample.txt', 'w') as f:
        # 初始化为0
        state = (torch.zeros(num_layers, 1, hidden_size).to(device),
                 torch.zeros(num_layers, 1, hidden_size).to(device))

        # 随机选择一个词作为输入
        prob = torch.ones(vocab_size)
        input = torch.multinomial(prob, num_samples=1).unsqueeze(1).to(device)

        for i in range(num_samples):
            # 从输入词开始，基于语言模型前推计算
            output, state = model(input, state)

            # 做预测
            prob = output.exp()
            word_id = torch.multinomial(prob, num_samples=1).item()

            # 填充预估结果（为下一次预估储备输入数据）
            input.fill_(word_id)

            # 写出输出结果
            word = corpus.dictionary.idx2word[word_id]
            word = '\n' if word == '<eos>' else word + ' '
            f.write(word)

            if (i+1) % 100 == 0:
                print('生成了 [{}/{}] 个词，存储到 {}'.format(i+1, num_samples, 'sample.txt'))

# 存储模型的保存点(checkpoints)
torch.save(model.state_dict(), 'model.ckpt')