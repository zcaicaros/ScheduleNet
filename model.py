from semiMDP.simulators import Simulator
import torch
import random
import numpy as np
from torch.nn.functional import relu, softmax
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax as pyg_softmax
from torch_geometric.nn.inits import reset
from torch.distributions.categorical import Categorical
from typing import Optional
from torch_geometric.typing import OptTensor


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

    input_graphs = {
        'pyg_assigned_agent': pyg_assigned_agent,
        'pyg_unassigned_agent': pyg_unassigned_agent,
        'pyg_assigned_task': pyg_assigned_task,
        'pyg_processable_task': pyg_processable_task,
        'pyg_unprocessable_task': pyg_unprocessable_task,
        'pyg_finished_task': pyg_finished_task
    }

    return input_graphs


class MLP(torch.nn.Module):
    def __init__(self,
                 num_layers=2,
                 in_chnl=8,
                 hidden_chnl=32,
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


class TGA_layer(MessagePassing):
    def __init__(
            self,
            # graphs parameters
            node_feature_num=11,
            k_dim=6,
            edge_feature_num=1,
            ## mlps parameters
            # layer
            etype_mlp_layer=1 + 1,
            edge_mlp_layer=2 + 1,
            attn_mlp_layer=2 + 1,
            ntype_mlp_layer=1 + 1,
            node_mlp_layer=2 + 1,
            # in dim
            etype_mlp_in_chnl=6,
            edge_mlp_in_chnl=32,
            attn_mlp_in_chnl=32,
            ntype_mlp_in_chnl=6,
            # hidden dim
            etype_mlp_hidden_chnl=32,
            edge_mlp_hidden_chnl=32,
            attn_mlp_hidden_chnl=32,
            ntype_mlp_hidden_chnl=32,
            node_mlp_hidden_chnl=32,
            # out dim
            etype_mlp_out_chnl=32,
            edge_mlp_out_chnl=32,
            attn_mlp_our_chnl=1,
            ntype_mlp_out_chnl=32,
            node_mlp_out_chnl=32,
            # MIs parameters
            out_dim_mi_node=32
    ):
        super(TGA_layer, self).__init__(node_dim=0, aggr='add')

        self.edge_h_updated = []

        self.etype_mlp = MLP(
            num_layers=etype_mlp_layer,
            in_chnl=etype_mlp_in_chnl,
            hidden_chnl=etype_mlp_hidden_chnl,
            out_chnl=etype_mlp_out_chnl
        )

        self.ntype_mlp = MLP(
            num_layers=ntype_mlp_layer,
            in_chnl=ntype_mlp_in_chnl,
            hidden_chnl=ntype_mlp_hidden_chnl,
            out_chnl=ntype_mlp_out_chnl
        )

        self.edge_mlp = MLP(
            num_layers=edge_mlp_layer,
            in_chnl=edge_mlp_in_chnl,
            hidden_chnl=edge_mlp_hidden_chnl,
            out_chnl=edge_mlp_out_chnl
        )

        self.node_mlp = MLP(
            num_layers=node_mlp_layer,
            in_chnl=node_feature_num + out_dim_mi_node,
            hidden_chnl=node_mlp_hidden_chnl,
            out_chnl=node_mlp_out_chnl
        )

        self.mi_edge = MI(
            ctx_size=etype_mlp_out_chnl,
            input_size=node_feature_num * 2 + edge_feature_num,
            output_size=edge_mlp_in_chnl
        )

        self.mi_node = MI(
            ctx_size=ntype_mlp_out_chnl,
            input_size=32,
            output_size=out_dim_mi_node
        )

        self.attn_mlp = MLP(
            num_layers=attn_mlp_layer,
            in_chnl=attn_mlp_in_chnl,
            hidden_chnl=attn_mlp_hidden_chnl,
            out_chnl=attn_mlp_our_chnl
        )

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.etype_mlp)
        reset(self.ntype_mlp)
        reset(self.edge_mlp)
        reset(self.node_mlp)
        reset(self.mi_edge)
        reset(self.mi_node)
        reset(self.attn_mlp)

    def message(
            self,
            x_j: Tensor,
            x_i: Tensor,
            edge_attr: OptTensor,
            index: Tensor,
            ptr: OptTensor,
            size_i: Optional[int]
    ) -> Tensor:

        k_i, k_j = x_i[:, :6], x_j[:, :6]
        h_i, h_j = x_i[:, 6:], x_j[:, 6:]

        c_j = self.etype_mlp(k_j)
        u_j = self.mi_edge(x=torch.cat([h_i, h_j, edge_attr], dim=1), z=c_j)
        h_j_prime = self.edge_mlp(u_j)
        self.edge_h_updated.append(h_j_prime)
        z_j = self.attn_mlp(u_j).squeeze()
        alpha_j = pyg_softmax(z_j, index, ptr, size_i)
        return h_j_prime * alpha_j.unsqueeze(-1)

    def forward(self, **graphs):
        self.edge_h_updated = []
        embeddings = []
        for name, graph in graphs.items():
            if graph.num_edges != 0:  # type k has edges
                # nodes with no neighbourhood get embedding = 0
                embedding = self.propagate(edge_index=graph.edge_index, x=graph.x, edge_attr=graph.edge_attr)
                embeddings.append(embedding)
            else:
                self.edge_h_updated.append(None)
        node_h_merged_by_sum = torch.stack(embeddings).sum(dim=0)

        x_i = graphs['pyg_assigned_agent'].x  # any graph will do
        h_i = x_i[:, 6:]
        k_i = x_i[:, :6]
        c_i = self.ntype_mlp(k_i)
        u_i = self.mi_node(node_h_merged_by_sum, c_i)
        h_i_prime = self.node_mlp(torch.cat([h_i, u_i], dim=1))

        # grad = torch.autograd.grad(h_i_prime.mean(), [param for param in self.parameters()])

        new_graphs = {}
        for i, (name, graph) in enumerate(graphs.items()):
            x_new_graph = torch.cat([graph.x[:, :6], h_i_prime], dim=1)
            if graph.num_edges != 0:  # type k has edges
                new_graphs[name] = Data(x=x_new_graph, edge_index=graph.edge_index, edge_attr=self.edge_h_updated[i]).to(graph.x.device)
            else:
                new_graphs[name] = Data(x=x_new_graph, edge_index=graph.edge_index, edge_attr=graph.edge_attr).to(graph.x.device)

        return new_graphs


class TGA(torch.nn.Module):
    def __init__(self):
        super(TGA, self).__init__()

        self.l1 = TGA_layer()
        self.l2 = TGA_layer(node_feature_num=32, edge_feature_num=32)

    def forward(self, **graphs):
        graphs = self.l1(**graphs)
        graphs = self.l2(**graphs)
        h_node = list(graphs.values())[0].x[:, 6:]  # any graph will do
        edge_index_merged = []
        edge_attr_merged = []
        for i, (name, graph) in enumerate(graphs.items()):
            if graph.num_edges != 0:
                edge_index_merged.append(graph.edge_index)
                edge_attr_merged.append(graph.edge_attr)
        edge_index_merged = torch.cat(edge_index_merged, dim=1)
        edge_attr_merged = torch.cat(edge_attr_merged, dim=0)
        merged_g = Data(x=h_node, edge_index=edge_index_merged, edge_attr=edge_attr_merged)
        return merged_g


class Policy(torch.nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

        self.mlp = MLP(
            num_layers=3,
            in_chnl=32 + 32 + 32,
            hidden_chnl=128,
            out_chnl=1
        )

    def forward(self, idle_machine, doable_op_id, embedded_g):
        edge_index_np = embedded_g.edge_index.cpu().numpy()
        idle_machine_flattened = np.array(idle_machine).repeat(repeats=[len(doable_ops_m) for doable_ops_m in doable_op_id])
        doable_op_id_flattened = np.array([op for ops_m in doable_op_id for op in ops_m])
        op_machine_pair = np.stack([doable_op_id_flattened, idle_machine_flattened])
        find_op = np.isin(edge_index_np[0], op_machine_pair[0])
        find_machine = np.isin(edge_index_np[1], op_machine_pair[1])
        which_edge = np.where(np.logical_and(find_op, find_machine))[0]
        # isin fn disturb the order of idle_machine_id and doable_op_id, we need rearrange op-machine pair
        _op_machine_pair_rearrange = edge_index_np[:, which_edge]

        h_i = embedded_g.x[_op_machine_pair_rearrange[1]]
        h_j = embedded_g.x[_op_machine_pair_rearrange[0]]
        h_ij = embedded_g.edge_attr[which_edge]

        pi = softmax(self.mlp(torch.cat([h_i, h_j, h_ij], dim=1)), dim=0).squeeze()
        dist = Categorical(probs=pi)
        sampled_op_id = dist.sample()
        sampled_op = _op_machine_pair_rearrange[0][sampled_op_id.item()]
        log_prob = dist.log_prob(sampled_op_id)

        return sampled_op, log_prob


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

    input_graphs = nx_to_pyg(g, dev)

    ctx = torch.rand(size=[3, 2])
    x = torch.rand(size=[3, 4])
    mi = MI(ctx_size=ctx.shape[1], input_size=x.shape[1], output_size=16)
    y = mi(x, ctx)

    tgae_l1 = TGA_layer().to(dev)
    out_graphs = tgae_l1(**input_graphs)
    # print(out_graphs)

    tgae_l2 = TGA_layer(
        node_feature_num=32,
        edge_feature_num=32,
    ).to(dev)
    # print(list(out_graphs.values())[0].edge_attr)
    out_graphs = tgae_l2(**out_graphs)
    # print(out_graphs)

    # test TGA net
    tga = TGA().to(dev)
    g = tga(**input_graphs)
    # grad = torch.autograd.grad(g.x.mean(), [param for param in tga.parameters()])

    # test policy
    policy = Policy().to(dev)
    op, log_p = policy([0, 2], [[9], [3, 6]], g)
    grad = torch.autograd.grad(log_p.mean(), [param for param in tga.parameters()])

