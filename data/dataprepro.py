
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from dgl.data.utils import generate_mask_tensor, idx2mask


def split_idx(samples, train_size, val_size, random_state=None):
    """Split the samples into training set, validation set and test set. 
    
    Float type satisfy: 
    * 0 < train_size < 1
    * 0 < val_size < 1
    * train_size + val_size < 1

    :param samples: list/ndarray/tensor 
    :param train_size: (int or float) If it is an integer, it represents the absolute 
    number of training samples; otherwise, it means the proportion of training samples 
    to all samples.
    :param val_size: (int or float) If it is an integer, it represents the absolute 
    number of validation samples; otherwise, it means the proportion of validation samples 
    to all samples.
    :param random_state: int, optional 
    :return: (train, val, test), same type as samples
    """
    train, val = train_test_split(samples, train_size=train_size, random_state=random_state)
    if isinstance(val_size, float):
        val_size *= len(samples) / len(val)
    val, test = train_test_split(val, train_size=val_size, random_state=random_state)
    return train, val, test

def get_label(degress, num_nodes, num_classes):

    labels = torch.zeros(num_nodes,dtype=int)
    splst = torch.div(degress.max()-degress.min(),num_classes)
    for i in range(num_classes):
        if i == num_classes-1:
            labels[np.where(degress>(splst*i))[0]] = int(i)
        else:
            labels[np.where((degress>(splst*i)) & (degress<(splst*(i+1))))[0]] = int(i)
    
    return labels

def get_node_mask(num, seed):
    train_idx, val_idx, test_idx = split_idx(np.arange(num), 0.2, 0.1, seed)
    train_mask = generate_mask_tensor(idx2mask(train_idx, num)).bool()
    val_mask = generate_mask_tensor(idx2mask(val_idx, num)).bool()
    test_mask = generate_mask_tensor(idx2mask(test_idx, num)).bool()
    return train_mask, val_mask, test_mask
