import torch
import numpy as np
import os

def parameter_num(net):
    total_num= sum(p.numel() for p in net.parameters())
    trainable_num= sum(p.numel() for p in net.parameters() if p.requires_grad)
    print({'Total': total_num, 'Trainable': trainable_num})

def cosine_learning_schedule(epochs, lr):
    cosine_learning_schedule= []

    for epoch in range(epochs):
        cos_inner= np.pi*(epoch%epochs)
        cos_inner/=epochs
        cos_out= np.cos(cos_inner)+1
        final_lr= float(lr/2*cos_out)
        cosine_learning_schedule.append(final_lr)
    return cosine_learning_schedule

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr']=lr

def set_seed(seed=15):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED']= str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic= True
