import torch
import const


torch.manual_seed(const.seed)

use_cuda = torch.cuda.is_available() and const.cuda_able
epochs = 1000
save = "BiLSTMresult/Bilstm.pt"
###################################################
#load data

from data_loader import DataLoader

data = torch.load(const.data)
max_len = data["max_len"]
vocab_size = data['dict']['vocab_size']
output_size = data['dict']['label_size']


training_data = DataLoader(data['train']['src'],
                           data['train']['label'],
                           max_len,
                           batch_size=const.batch_size,
                           cuda=use_cuda)
validation_data = DataLoader(data['valid']['src'],
                             data['valid']['label'],
                             max_len,
                             batch_size=const.batch_size,
                             shuffle=False,
                             cuda=use_cuda)


# 模型
import model

BiLSTM_model = model.BiLSTM_Attn(vocab_size = vocab_size,
                                 emb_size=const.biLSTM_param["embed_dim"],
                                 lstm_hidden_size=const.biLSTM_param["hidden_size"],
                                 output_size = output_size,
                                 batch_size= const.batch_size,
                                 bidirectional=True,
                                 dropout=const.dropout,
                                 use_cuda=use_cuda,
                                 attention_size=const.biLSTM_param["attention_size"],
                                 sequence_length=const.biLSTM_param["sequence_length"]
                                 )
import textCNN

if use_cuda:
    lstm_attn = BiLSTM_model.cuda()


optimizer = torch.optim.Adam(lstm_attn.parameters(), lr=const.lr, weight_decay=const.weight_decay)
criterion = torch.nn.CrossEntropyLoss()

###################################################
#training
import time
from tqdm import tqdm

train_loss = []
valid_loss = []
accuracy = []


def train(current_model):
    current_model.train()
    total_loss = 0.0
    for data, label in tqdm(training_data, mininterval=1,
                            desc='Train Processing', leave=False):
        data = data.long()
        label = label.long()
        optimizer.zero_grad()
        out = current_model(data)
        loss = criterion(out,label)
        loss.backward()
        optimizer.step()
        total_loss += loss.data
    return total_loss.item() / training_data.sents_size

def evaluate(current_model):
    current_model.eval()
    corrects = eval_loss = 0
    _size = validation_data.sents_size

    for data, label in tqdm(validation_data, mininterval=0.2,
                            desc='Evaluate Processing', leave=False):
        data = data.long()
        label = label.long()
        pred = lstm_attn(data)
        loss = criterion(pred, label)

        eval_loss += loss.data
        corrects += (torch.max(pred, 1)[1].view(label.size()).data == label.data).sum()
    return eval_loss.item() / _size, corrects, corrects * 100.0 / _size, _size
#################################################
#saving
best_acc = None
total_start_time = time.time()

try:
    print('-' * 90)
    for epoch in range(1, epochs+1):
        epoch_start_time = time.time()
        loss = train(BiLSTM_model)
        train_loss.append(loss*1000.)

        print('| start of epoch {:3d} | time: {:2.2f}s | loss {:5.6f}'.format(epoch,
                                                                              time.time() - epoch_start_time,
                                                                              loss))

        loss, corrects, acc, size = evaluate(BiLSTM_model)
        valid_loss.append(loss*1000.)
        accuracy.append(acc)

        print('-' * 10)
        print('| end of epoch {:3d} | time: {:2.2f}s | loss {:.4f} | accuracy {}%({}/{})'.format(epoch,
                                                                                                 time.time() - epoch_start_time,
                                                                                                 loss,
                                                                                                 acc,
                                                                                                 corrects,
                                                                                                 size))
        print('-' * 10)
        if not best_acc or best_acc < corrects:
            best_acc = corrects
            model_state_dict = BiLSTM_model.state_dict()
            model_source = {
                "model": model_state_dict,
                "src_dict": data['dict']['train']
            }
            torch.save(model_source, save)
except KeyboardInterrupt:
    print("-"*90)
    print("Exiting from training early | cost time: {:5.2f}min".format((time.time() - total_start_time)/60.0))