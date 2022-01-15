from semiMDP.simulators import Simulator
import torch
import random
import numpy as np
import networkx as nx
from torch.nn.functional import relu, softmax
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch.distributions.categorical import Categorical


def one_hotter(arr):
    """
    arr: [n,] ndarray
    """
    n_values = np.max(arr) + 1
    return np.eye(n_values, dtype=np.float32)[arr]


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


def nx_to_pyg(g, dev):

    node_feature_all = []
    k_all = []
    for n in g.nodes:
        node_feature_all.append(
            [g.nodes[n]['agent'],
             g.nodes[n]['target_agent'],
             g.nodes[n]['assigned'],
             g.nodes[n]['waiting'],
             g.nodes[n]['processable'],
             g.nodes[n]['accessible'],
             g.nodes[n]['task_wait_time'],
             g.nodes[n]['task_processing_time'],
             g.nodes[n]['time_to_complete'],
             g.nodes[n]['remain_ops'],
             g.nodes[n]['job_completion_ratio']]
        )
        if g.nodes[n]['node_type'] == 'assigned_agent':
            k_all.append(0)
        elif g.nodes[n]['node_type'] == 'unassigned_agent':
            k_all.append(1)
        elif g.nodes[n]['node_type'] == 'assigned_task':
            k_all.append(2)
        elif g.nodes[n]['node_type'] == 'processable_task':
            k_all.append(3)
        elif g.nodes[n]['node_type'] == 'unprocessable_task':
            k_all.append(4)
        elif g.nodes[n]['node_type'] == 'completed_task':
            k_all.append(5)
        else:
            raise RuntimeError("Not supported node type.")

    node_feature_all = np.array(node_feature_all, dtype=np.float32)
    k_all = one_hotter(np.array(k_all)).reshape(g.number_of_nodes(), -1)

    adj_assigned_agent = np.zeros([g.number_of_nodes(), g.number_of_nodes()], dtype=np.float32)
    adj_unassigned_agent = np.zeros([g.number_of_nodes(), g.number_of_nodes()], dtype=np.float32)
    adj_assigned_task = np.zeros([g.number_of_nodes(), g.number_of_nodes()], dtype=np.float32)
    adj_processable_task = np.zeros([g.number_of_nodes(), g.number_of_nodes()], dtype=np.float32)
    adj_unprocessable_task = np.zeros([g.number_of_nodes(), g.number_of_nodes()], dtype=np.float32)
    adj_finished_task = np.zeros([g.number_of_nodes(), g.number_of_nodes()], dtype=np.float32)

    edge_attr_assigned_agent = []
    edge_attr_unassigned_agent = []
    edge_attr_assigned_task = []
    edge_attr_processable_task = []
    edge_attr_unprocessable_task = []
    edge_attr_finished_task = []
    for e in g.edges:
        s, t = e
        if g.nodes[s]['node_type'] == 'assigned_agent':
            adj_assigned_agent[s, t] = 1
            edge_attr_assigned_agent.append(g.get_edge_data(*e)['edge_feature'])
        elif g.nodes[s]['node_type'] == 'unassigned_agent':
            adj_unassigned_agent[s, t] = 1
            edge_attr_unassigned_agent.append(g.get_edge_data(*e)['edge_feature'])
        elif g.nodes[s]['node_type'] == 'assigned_task':
            adj_assigned_task[s, t] = 1
            edge_attr_assigned_task.append(g.get_edge_data(*e)['edge_feature'])
        elif g.nodes[s]['node_type'] == 'processable_task':
            adj_processable_task[s, t] = 1
            edge_attr_processable_task.append(g.get_edge_data(*e)['edge_feature'])
        elif g.nodes[s]['node_type'] == 'unprocessable_task':
            adj_unprocessable_task[s, t] = 1
            edge_attr_unprocessable_task.append(g.get_edge_data(*e)['edge_feature'])
        elif g.nodes[s]['node_type'] == 'completed_task':
            adj_finished_task[s, t] = 1
            edge_attr_finished_task.append(g.get_edge_data(*e)['edge_feature'])
        else:
            raise RuntimeError("Error, contains Not supported neighbourhood type {}.".format(g.nodes[s]['node_type']))

    # create pyg sub-graphs for each neighbourhood type
    x = torch.from_numpy(np.concatenate([k_all, node_feature_all], axis=1))
    pyg_assigned_agent = Data(
        x=x,
        edge_index=torch.nonzero(torch.from_numpy(adj_assigned_agent)).t().contiguous(),
        edge_attr=torch.from_numpy(np.array(edge_attr_assigned_agent, dtype=np.float32))
    )
    pyg_unassigned_agent = Data(
        x=x,
        edge_index=torch.nonzero(torch.from_numpy(adj_unassigned_agent)).t().contiguous(),
        edge_attr=torch.from_numpy(np.array(edge_attr_unassigned_agent, dtype=np.float32))
    )
    pyg_assigned_task = Data(
        x=x,
        edge_index=torch.nonzero(torch.from_numpy(adj_assigned_task)).t().contiguous(),
        edge_attr=torch.from_numpy(np.array(edge_attr_assigned_task, dtype=np.float32))
    )
    pyg_processable_task = Data(
        x=x,
        edge_index=torch.nonzero(torch.from_numpy(adj_processable_task)).t().contiguous(),
        edge_attr=torch.from_numpy(np.array(edge_attr_processable_task, dtype=np.float32))
    )
    pyg_unprocessable_task = Data(
        x=x,
        edge_index=torch.nonzero(torch.from_numpy(adj_unprocessable_task)).t().contiguous(),
        edge_attr=torch.from_numpy(np.array(edge_attr_unprocessable_task, dtype=np.float32))
    )
    pyg_finished_task = Data(
        x=x,
        edge_index=torch.nonzero(torch.from_numpy(adj_finished_task)).t().contiguous(),
        edge_attr=torch.from_numpy(np.array(edge_attr_finished_task, dtype=np.float32))
    )
    # to device
    pyg_assigned_agent.to(dev)
    pyg_unassigned_agent.to(dev)
    pyg_assigned_task.to(dev)
    pyg_processable_task.to(dev)
    pyg_unprocessable_task.to(dev)
    pyg_finished_task.to(dev)

    return pyg_assigned_agent, pyg_unassigned_agent, pyg_assigned_task, pyg_processable_task, pyg_unprocessable_task, pyg_finished_task


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

    nx_to_pyg(g, dev)