import torch
import const
import torch.nn as nn
from const import *
import tqdm
from textCNN import TextCNN
torch.manual_seed(const.seed)
import sys
import pytorch_lightning
from pytorch_lightning.metrics.functional import accuracy
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
model = TextCNN(vocab_size=vocab_size,emb_dim=textCNN_param["embed_dim"],filter_sizes=textCNN_param["filter_size"],
                num_filter=textCNN_param["filter_num"],num_classes=output_size,embedding_pretrained=None,dropout=0.4)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(),lr=lr)
def train():
    if use_cuda:
        model.cuda()
    model.train()
    steps = 0
    best_acc = 0
    last_step = 0
    train_loss = 0.0
    print("start training")
    for epoch in range(1,epochs+1):
        for data, label in training_data:
            data = data.long().cuda()
            label = label.long().cuda()

            predict = model(data)
            optimizer.zero_grad()
            loss = criterion(predict,label)
            loss.backward()
            optimizer.step()

            steps+=1
            if steps%50==0:
                #result = torch.max(predict,1)[1].view(label.size())
                #corrects = (result.data==label.data).sum()
                #acc = corrects*100.0/batch_size
                pre = torch.softmax(predict.data,dim=1)
                ret,predictions = torch.max(pre.data,1)
                acc = accuracy(predictions,label)
                sys.stdout.write('\rtraining：Batch[{}] - current_loss: {:.6f} acc: {:.4f}'.format(steps,
                                                                                         loss.data.item(),
                                                                                         acc,
                                                                                        ))
            if steps%100==0:
                dev_acc = eval(model,steps)
                if dev_acc>best_acc:
                    best_acc=dev_acc
                    last_step = steps
                    # torch.save()
def eval(model,steps):
    model.eval()
    avg_loss = 0.0
    with torch.no_grad():
        for data, label in validation_data:
            data = data.long().cuda()
            label = label.long().cuda()

            predict = model(data)
            loss = criterion(predict,label)
            pre = torch.softmax(predict,1)
            avg_loss+=loss.item()
            ret,predictions = torch.max(pre.data,1)
            acc  = accuracy(predictions,label)
            avg_loss/=batch_size
    print('\nEvaluation - loss: {:.6f} acc: {:.4f} \n'.format(avg_loss,acc))

    return  acc

train()