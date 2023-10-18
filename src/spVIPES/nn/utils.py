import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init


def one_hot(index: torch.Tensor, n_cat: int) -> torch.Tensor:
    """One hot a tensor of categories."""
    onehot = torch.zeros(index.size(0), n_cat, device=index.device)
    onehot.scatter_(1, index.type(torch.long), 1)
    return onehot.type(torch.float32)


def kaiming_init(m, seed):
    """
    Initialize the parameters of a PyTorch model using the Kaiming initialization method.

    Parameters
    ----------
        m (nn.Module): The PyTorch module for which to initialize the parameters.
        seed (int): Random seed for initialization.

    This function sets the weights and biases of the provided PyTorch module `m` using Kaiming initialization.
    It also sets random seeds for reproducibility.

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)
