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


def to_pyg(g, dev):
    assigned_agents = []
    unassigned_agents = []
    assigned_tasks = []
    processable_tasks = []
    unprocessable_tasks = []
    # aggregate node according to node type
    for n in g.nodes:
        if g.nodes[n]['node_type'] == 'assigned_agent':
            assigned_agents.append(n)
        if g.nodes[n]['node_type'] == 'unassigned_agent':
            unassigned_agents.append(n)
        if g.nodes[n]['node_type'] == 'assigned_task':
            assigned_tasks.append(n)
        if g.nodes[n]['node_type'] == 'processable_task':
            processable_tasks.append(n)
        if g.nodes[n]['node_type'] == 'unprocessable_task':
            unprocessable_tasks.append(n)
        # print(n, g.nodes[n])

    assigned_agents_induced_g = g.subgraph(assigned_agents)
    unassigned_agents_induced_g = g.subgraph(unassigned_agents)
    assigned_tasks_induced_g = g.subgraph(assigned_tasks)
    processable_tasks_induced_g = g.subgraph(processable_tasks)
    unprocessable_tasks_induced_g = g.subgraph(unprocessable_tasks)
    
    pass


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(1)
    torch.manual_seed(1)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    # dev = 'cpu'

    s = Simulator(3, 3, verbose=False)
    print(s.machine_matrix)
    print(s.processing_time_matrix)
    s.reset()

    g, r, done = s.observe()

    to_pyg(g, dev)