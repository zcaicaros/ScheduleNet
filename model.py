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


def one_hotter(arr, n_values=6):
    """
    arr: [n,] ndarray
    """
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
    print(k_all)

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


class MLP(torch.nn.Module):
    def __init__(self,
                 num_layers=2,
                 in_chnl=8,
                 hidden_chnl=256,
                 out_chnl=8):
        super(MLP, self).__init__()

        self.layers = torch.nn.ModuleList()

        for l in range(num_layers):
            if l == 0:  # first layer
                self.layers.append(torch.nn.Linear(in_chnl, hidden_chnl))
                self.layers.append(torch.nn.ReLU())
                if num_layers == 1:
                    self.layers.append(torch.nn.Linear(hidden_chnl, out_chnl))
            elif l <= num_layers - 2:  # hidden layers
                self.layers.append(torch.nn.Linear(hidden_chnl, hidden_chnl))
                self.layers.append(torch.nn.ReLU())
            else:  # last layer
                self.layers.append(torch.nn.Linear(hidden_chnl, hidden_chnl))
                self.layers.append(torch.nn.ReLU())
                self.layers.append(torch.nn.Linear(hidden_chnl, out_chnl))

    def forward(self, h):
        for lyr in self.layers:
            h = lyr(h)
        return h


class MI(torch.nn.Module):
    def __init__(self, ctx_size, input_size, output_size):
        super(MI, self).__init__()

        self.W = torch.nn.Linear(ctx_size, input_size * output_size)
        self.b = torch.nn.Linear(ctx_size, output_size)

    def forward(self, x, z):
        """
        :param x: [B, input size]
        :param z: [B, context size]
        :return: y: [B, output_size]
        """
        W_prime = self.W(z).reshape(x.shape[0], x.shape[1], -1)
        b_prime = self.b(z)
        y = torch.matmul(x.unsqueeze(1), W_prime).squeeze() + b_prime

        return y


class TGAe_layer(MessagePassing):
    def __init__(self,
                 etype_mlp_layer=1,
                 edge_mlp_layer=1,
                 etype_mlp_in_chnl=6,
                 edge_mlp_in_chnl=1,
                 node_feature_num=11,
                 edge_feature_num=1,
                 etype_mlp_hidden_chnl=32,
                 edge_mlp_hidden_chnl=32,
                 etype_mlp_out_chnl=32,
                 edge_mlp_out_chnl=32):
        super(TGAe_layer, self).__init__()

        self.etype_mlp = MLP(
            num_layers=etype_mlp_layer,
            in_chnl=etype_mlp_in_chnl,
            hidden_chnl=etype_mlp_hidden_chnl,
            out_chnl=etype_mlp_out_chnl
        )

        self.edge_mlp = MLP(
            num_layers=edge_mlp_layer,
            in_chnl=edge_mlp_in_chnl,
            hidden_chnl=edge_mlp_hidden_chnl,
            out_chnl=edge_mlp_out_chnl
        )

        self.MI = MI(
            ctx_size=etype_mlp_out_chnl,
            input_size=node_feature_num * 2 + edge_feature_num,
            output_size=32
        )

    def forward(self, **graphs):
        for name, graph in graphs.items():
            if graph.num_edges != 0:
                kj = graph.x[graph.edge_index[0]][:, :6]
                hi = graph.x[graph.edge_index[1]][:, 6:]
                hj = graph.x[graph.edge_index[0]][:, 6:]
                hij = graph.edge_attr
                cij = self.etype_mlp(kj)
                uij = self.MI(x=torch.cat([hi, hj, hij], dim=1), z=cij)


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

    pyg_assigned_agent, pyg_unassigned_agent, pyg_assigned_task, pyg_processable_task, pyg_unprocessable_task, pyg_finished_task = nx_to_pyg(g, dev)

    ctx = torch.rand(size=[3, 2])
    x = torch.rand(size=[3, 4])
    mi = MI(ctx_size=ctx.shape[1], input_size=x.shape[1], output_size=16)
    y = mi(x, ctx)
    # print(y.shape)

    tgae = TGAe_layer().to(dev)
    input_graphs = {
        'pyg_assigned_agent': pyg_assigned_agent,
        'pyg_unassigned_agent': pyg_unassigned_agent,
        'pyg_assigned_task': pyg_assigned_task,
        'pyg_processable_task': pyg_processable_task,
        'pyg_unprocessable_task': pyg_unprocessable_task,
        'pyg_finished_task': pyg_finished_task
    }
    tgae(**input_graphs)