import torch
import torch.nn.functional as F

def gumbel_sample(shape,eps=1e-20):
    U = torch.rand(shape)
    U = U.cuda()
    return -torch.log(-torch.log(U+eps)+eps)

def gumbel_softmax_sample(logits,temperature=1.0):
    y = logits+gumbel_sample(logits.size())

    return F.softmax(y/temperature,dim=-1)

def gumbel_softmax(logits,temperature=1.0,hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits,temperature)

    if not hard:
        return y

    shape = y.size()
    _,ind = y.max(dim=1)
    y_hard = torch.zeros_like(y).view(-1,shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    # 返回的结果是采样结果y_hard，而反向传播求导时却是对y求到的
    y_hard = (y_hard - y).detach() + y
    return y_hard