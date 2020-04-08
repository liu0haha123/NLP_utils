import torch
from torch import autograd
from math import ceil
from torch.autograd import Variable
def prepare_generator_batch(samples, start_letter=0, gpu=False):
    # 从采样序列中返回训练数据与标签
    batch_size,seq_len = samples.size()

    inputs = torch.zeros(batch_size,seq_len)
    targets = samples
    inputs[:,0] = start_letter
    inputs[:,1:] = targets[:,:seq_len-1]

    inputs = Variable(inputs).type(torch.LongTensor)
    targets = Variable(targets).type(torch.LongTensor)

    if gpu:
        inp = inputs.cuda()
        target = targets.cuda()

    return inp, target

def prepare_discriminator_data(pos_samples, neg_samples, gpu=False):
    # 为判别器分别采样真实数据和生成数据
    # 正例：pos（pos_size,seq_len)负例neg（neg_size,seq_len)
    # 输出     - inp: (pos_size + neg_size) x seq_len
    #         - target: pos_size + neg_size (boolean 1/0)
    pos_size,batch_size  =pos_samples.size()
    neg_size,_ =neg_samples.size()
    inputs = torch.cat([pos_samples,neg_samples],dim=0).type(torch.LongTensor)
    targets = torch.ones(neg_size+pos_size)
    # 负例标签为0
    targets[pos_size:] = 0
    # shuffle
    perm = torch.randperm(targets.size()[0])
    targets = targets[perm]
    inputs = inputs[perm]

    inputs = Variable(inputs)
    targets = Variable(targets)

    if gpu:
        inp = inputs.cuda()
        target = targets.cuda()

    return inputs, targets


def batchwise_sample(gen, num_samples, batch_size):
    """
    按照batch采样
    """

    samples = []
    for i in range(int(ceil(num_samples/float(batch_size)))):
        samples.append(gen.sample(batch_size))

    return torch.cat(samples, 0)[:num_samples]


def batchwise_oracle_nll(gen, oracle, num_samples, batch_size, max_seq_len, start_letter=0, gpu=False):

    # 计算NLLLoss
    s = batchwise_sample(gen, num_samples, batch_size)
    oracle_nll = 0
    for i in range(0, num_samples, batch_size):
        inp, target = prepare_generator_batch(s[i:i+batch_size], start_letter, gpu)
        oracle_loss = oracle.batchNLLLoss(inp, target) / max_seq_len
        oracle_nll += oracle_loss.data.item()

    return oracle_nll/(num_samples/batch_size)
