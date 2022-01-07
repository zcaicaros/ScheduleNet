from semiMDP.simulators import Simulator
import torch
import random
import numpy as np
import networkx as nx
from torch.nn.functional import relu, softmax
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch.distributions.categorical import Categorical


def count_parameters(model, verbose=False, print_model=False):
    """
    model: torch nn
    """
    if print_model:
        print('Model:', model)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.data)

    print('The model has {} parameters'.format(pytorch_total_params))


