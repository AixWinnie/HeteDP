import torch
import dgl
import random
import numpy as np
from sklearn.metrics import f1_score

def set_random_seed(seed):
    """Set the random seed of Python, numpy and PyTorch

    :param seed: int
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    dgl.seed(seed)


def get_device(device):
    """Returns the specified GPU device

    :param device: (int) GPU number, -1 for CPU
    :return: torch.device
    """
    return torch.device(f'cuda:{device}' if device >= 0 and torch.cuda.is_available() else 'cpu')

def accuracy(logits, labels):
    """Calculate accuracy

    :param logits: tensor(N, C) is predicted probability, N is samples number, C is categories number
    :param labels: tensor(N) 
    :return: float 
    """
    return torch.sum(torch.argmax(logits, dim=1) == labels).item() * 1.0 / len(labels)


def micro_macro_f1_score(logits, labels):
    """Calculate Micro-F1 and Macro-F1 score

    :param logits: tensor(N, C) is predicted probability, N is samples number, C is categories number
    :param labels: tensor(N) 
    :return: float
    """
    prediction = torch.argmax(logits, dim=1).cpu().long().numpy()
    labels = labels.cpu().numpy() 
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')
    return micro_f1, macro_f1

class dataset():
    def __init__(self, args):
        self.hidden = args.hidden_topo
        self.out = args.out_topo
        self.delta = args.delta
        self.eps_requirement = args.epsilon_topo
        self.noise_sigma = 2.0
        self.batch_proc_size = args.batchsize
        self.grad_norm_max = 1
        self.epochs = args.epochs
        self.C_decay = 0.95
        
    def forward(self, g):
        self.sample_rate = self.batch_proc_size / g.num_nodes()
        
def load_config(args, g):
    data = dataset(args)
    data.forward(g)
    return data
