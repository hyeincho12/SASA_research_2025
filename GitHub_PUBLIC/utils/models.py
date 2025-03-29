import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, AttentiveFP
from torch_geometric.nn import global_add_pool



# Thomas N. Kipf and Max Welling,
# Semi-Supervised Classification with Graph Convolutional Networks
# International Conference on Learning Representations (ICLR) 2017
class GCN(nn.Module):
    # graph conovolutional network (GCN)
    def __init__(self, num_node_feats, n_mol_feats, dims, n_layers):
        super(GCN, self).__init__()
        self.n_layers = n_layers

        # graph convolution layers
        self.gc = nn.ModuleList()
        self.gc.append(GCNConv(num_node_feats, dims[0]))
        self.bn_gc = nn.BatchNorm1d(dims[0])
        if n_layers[0] > 2:
            for i in range(0, n_layers[0] - 2):
                self.gc.append(GCNConv(dims[0], dims[0]))
        self.gc.append(GCNConv(dims[0], dims[1]))

        # fully-connected layers for molecular features
        self.fc_m = nn.ModuleList()
        self.fc_m.append(nn.Linear(n_mol_feats, dims[0]))
        self.bn_m = nn.BatchNorm1d(dims[0])
        if n_layers[1] > 2:
            for i in range(0, n_layers[1] - 2):
                self.fc_m.append(nn.Linear(dims[0], dims[0]))
        self.fc_m.append(nn.Linear(dims[0], dims[1]))

        # fully-connected layers for all features
        self.fc = nn.ModuleList()
        self.fc.append(nn.Linear(dims[1] * 2, dims[2]))
        for i in range(0, n_layers[2] - 3):
            self.fc.append(nn.Linear(dims[2], dims[2]))
        self.fc.append(nn.Linear(dims[2], dims[3]))
        self.fc.append(nn.Linear(dims[3], dims[-1]))

    def forward(self, g):
        h = F.silu(self.bn_gc(self.gc[0](g.x, g.edge_index)))
        for i in range(1, self.n_layers[0]):
            h = F.silu(self.gc[i](h, g.edge_index))
        hg = global_add_pool(h, g.batch)

        h_m = F.silu(self.bn_m(self.fc_m[0](g.mol_feats)))
        for i in range(1, self.n_layers[1]):
            h_m = F.silu(self.fc_m[i](h_m))

        hg = torch.cat([hg, h_m], dim=1)
        for i in range(0, self.n_layers[2] - 1):
            hg = F.silu(self.fc[i](hg))
        out = self.fc[-1](hg)

        return out


class GCN_EN(nn.Module):
    # graph conovolutional network (GCN)
    def __init__(self, num_node_feats, n_mol_feats, dims, n_layers, en_dim):
        super(GCN_EN, self).__init__()
        self.n_layers = n_layers
        self.fc_en = nn.Linear(num_node_feats, en_dim)

        # graph convolution layers
        self.gc = nn.ModuleList()
        self.gc.append(GCNConv(en_dim, dims[0]))
        self.bn_gc = nn.BatchNorm1d(dims[0])
        if n_layers[0] > 2:
            for i in range(0, n_layers[0] - 2):
                self.gc.append(GCNConv(dims[0], dims[0]))
        self.gc.append(GCNConv(dims[0], dims[1]))

        # fully-connected layers for molecular features
        self.fc_m = nn.ModuleList()
        self.fc_m.append(nn.Linear(n_mol_feats, dims[0]))
        self.bn_m = nn.BatchNorm1d(dims[0])
        if n_layers[1] > 2:
            for i in range(0, n_layers[1] - 2):
                self.fc_m.append(nn.Linear(dims[0], dims[0]))
        self.fc_m.append(nn.Linear(dims[0], dims[1]))

        # fully-connected layers for all features
        self.fc = nn.ModuleList()
        self.fc.append(nn.Linear(dims[1] * 2, dims[2]))
        for i in range(0, n_layers[2] - 3):
            self.fc.append(nn.Linear(dims[2], dims[2]))
        self.fc.append(nn.Linear(dims[2], dims[3]))
        self.fc.append(nn.Linear(dims[3], dims[-1]))

    def forward(self, g):
        en = self.calc_en(g)

        h = F.silu(self.bn_gc(self.gc[0](en, g.edge_index)))
        for i in range(1, self.n_layers[0]):
            h = F.silu(self.gc[i](h, g.edge_index))
        hg = global_add_pool(h, g.batch)

        h_m = F.silu(self.bn_m(self.fc_m[0](g.mol_feats)))
        for i in range(1, self.n_layers[1]):
            h_m = F.silu(self.fc_m[i](h_m))

        hg = torch.cat([hg, h_m], dim=1)
        for i in range(0, self.n_layers[2] - 1):
            hg = F.silu(self.fc[i](hg))
        out = self.fc[-1](hg)

        return out

    def calc_en(self, g):
        # generate embedding vector at given dimension
        fc_out = self.fc_en(g.x)
        # make the norm to unity
        fc_out = F.normalize(fc_out, p=2, dim=1)
        # multiply the Pauling electronegativity
        en = g.atom_ens.view(-1, 1) * fc_out

        return en

    def prt_emb(self, g):
        en = self.calc_en(g)

        h = F.silu(self.bn_gc(self.gc[0](en, g.edge_index)))
        for i in range(1, self.n_layers[0]):
            h = F.silu(self.gc[i](h, g.edge_index))
        hg = global_add_pool(h, g.batch)

        h_m = F.silu(self.bn_m(self.fc_m[0](g.mol_feats)))
        for i in range(1, self.n_layers[1]):
            h_m = F.silu(self.fc_m[i](h_m))

        hg = torch.cat([hg, h_m], dim=1)
        for i in range(0, self.n_layers[2] - 1):
            hg = F.silu(self.fc[i](hg))

        return hg

# Zhaoping Xiong et al.
# Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph Attention Mechanism
# J. Med. Chem. 2020, 63, 16, 8749–8760
class AttFP(nn.Module):
    def __init__(self, num_node_feats, num_edge_feats, n_mol_feats, dims, n_layers):
        super(AttFP, self).__init__()
        self.n_layers = n_layers

        # graph convolution layers
        self.gc = nn.ModuleList()
        self.gc.append(AttentiveFP(in_channels=num_node_feats, hidden_channels=dims[0], out_channels=dims[1], edge_dim=num_edge_feats,
                                   num_timesteps=3, num_layers=n_layers[0], dropout=0.1))

        # fully-connected layers for molecular features
        self.fc_m = nn.ModuleList()
        self.fc_m.append(nn.Linear(n_mol_feats, dims[0]))
        if n_layers[1] > 2:
            for i in range(0, n_layers[1] - 2):
                self.fc_m.append(nn.Linear(dims[0], dims[0]))
        self.fc_m.append(nn.Linear(dims[0], dims[1]))

        # fully-connected layers for all features
        self.fc = nn.ModuleList()
        self.fc.append(nn.Linear(dims[1] * 2, dims[2]))
        for i in range(0, n_layers[2] - 3):
            self.fc.append(nn.Linear(dims[2], dims[2]))
        self.fc.append(nn.Linear(dims[2], dims[3]))
        self.fc.append(nn.Linear(dims[3], dims[-1]))

    def forward(self, g):
        hg = self.gc[0](g.x, g.edge_index, g.edge_attr, g.batch)
        h_m = F.silu(self.fc_m[0](g.mol_feats))
        for i in range(1, self.n_layers[1]):
            h_m = F.silu(self.fc_m[i](h_m))

        hg = torch.cat([hg, h_m], dim=1)
        for i in range(0, self.n_layers[2] - 1):
            hg = F.silu(self.fc[i](hg))
        out = self.fc[-1](hg)

        return out

    def prt_emb(self, g):
        hg = self.gc[0](g.x, g.edge_index, g.edge_attr, g.batch)
        h_m = F.silu(self.fc_m[0](g.mol_feats))
        for i in range(1, self.n_layers[1]):
            h_m = F.silu(self.fc_m[i](h_m))

        hg = torch.cat([hg, h_m], dim=1)
        for i in range(0, self.n_layers[2] - 1):
            hg = F.silu(self.fc[i](hg))

        return hg


class AttFP2(nn.Module):
    def __init__(self, num_node_feats, num_edge_feats, n_mol_feats, dims, n_layers):
        super(AttFP2, self).__init__()
        self.n_layers = n_layers

        # graph convolution layers
        self.gc = nn.ModuleList()
        self.gc.append(AttentiveFP(in_channels=num_node_feats, hidden_channels=dims[0], out_channels=dims[-1], edge_dim=num_edge_feats,
                                   num_timesteps=5, num_layers=n_layers[0], dropout=0.1))

    def forward(self, g):
        out = self.gc[0](g.x, g.edge_index, g.edge_attr, g.batch)

        return out

# Shaked Brody, Uri Alon, Eran Yahav
# How Attentive are Graph Attention Networks?
# https://arxiv.org/abs/2105.14491, ICLR 2022
class GATv2(nn.Module):
    def __init__(self, num_node_feats, num_edge_feats, n_mol_feats, multi_head, dims, n_layers):
        super(GATv2, self).__init__()
        self.n_layers = n_layers

        # graph convolution layers
        self.gc = nn.ModuleList()
        self.gc.append(GATv2Conv(in_channels=num_node_feats, out_channels=dims[0], edge_dim=num_edge_feats,
                                   heads=multi_head, concat=True))
        if n_layers[0] > 2:
            for i in range(0, n_layers[0] - 2):
                self.gc.append( GATv2Conv(in_channels=dims[0] * multi_head, out_channels=dims[0], edge_dim=num_edge_feats,
                                       heads=multi_head, concat=True))
        self.gc.append(GATv2Conv(in_channels=dims[0] * multi_head, out_channels=dims[1], edge_dim=num_edge_feats,
                                   heads=multi_head, concat=True))
        self.bn_gc = nn.BatchNorm1d(dims[0] * multi_head)

        # fully-connected layers for molecular features
        self.fc_m = nn.ModuleList()
        self.fc_m.append(nn.Linear(n_mol_feats, dims[0]))
        self.bn_m = nn.BatchNorm1d(dims[0])
        if n_layers[1] > 2:
            for i in range(0, n_layers[1] - 2):
                self.fc_m.append(nn.Linear(dims[0], dims[0]))
        self.fc_m.append(nn.Linear(dims[0], dims[1]))

        # fully-connected layers for all features
        self.fc = nn.ModuleList()
        self.fc.append(nn.Linear(dims[1] * multi_head + dims[1], dims[2]))
        for i in range(0, n_layers[2] - 3):
            self.fc.append(nn.Linear(dims[2], dims[2]))
        self.fc.append(nn.Linear(dims[2], dims[3]))
        self.fc.append(nn.Linear(dims[3], dims[-1]))

    def forward(self, g):
        h = F.silu(self.bn_gc(self.gc[0](g.x, g.edge_index)))
        for i in range(1, self.n_layers[0]):
            h = F.silu(self.gc[i](h, g.edge_index))
        hg = global_add_pool(h, g.batch)

        h_m = F.silu(self.bn_m(self.fc_m[0](g.mol_feats)))
        for i in range(1, self.n_layers[1]):
            h_m = F.silu(self.fc_m[i](h_m))

        hg = torch.cat([hg, h_m], dim=1)
        for i in range(0, self.n_layers[2] - 1):
            hg = F.silu(self.fc[i](hg))
        out = self.fc[-1](hg)

        return out

    def prt_emb(self, g):
        h = F.silu(self.bn_gc(self.gc[0](g.x, g.edge_index)))
        for i in range(1, self.n_layers[0]):
            h = F.silu(self.gc[i](h, g.edge_index))
        hg = global_add_pool(h, g.batch)

        h_m = F.silu(self.bn_m(self.fc_m[0](g.mol_feats)))
        for i in range(1, self.n_layers[1]):
            h_m = F.silu(self.fc_m[i](h_m))

        hg = torch.cat([hg, h_m], dim=1)
        for i in range(0, self.n_layers[2] - 1):
            hg = F.silu(self.fc[i](hg))

        return hg


class GATv2AFP(nn.Module):
    def __init__(self, num_node_feats, num_edge_feats, n_mol_feats, multi_head, dims, n_layers):
        super(GATv2AFP, self).__init__()
        self.n_layers = n_layers

        # graph convolution layers
        self.gc = nn.ModuleList()
        self.gc.append(GATv2Conv(in_channels=num_node_feats, out_channels=dims[0], edge_dim=num_edge_feats,
                                   heads=multi_head, concat=True))
        if n_layers[0] > 2:
            for i in range(0, n_layers[0] - 2):
                self.gc.append( GATv2Conv(in_channels=dims[0] * multi_head, out_channels=dims[0], edge_dim=num_edge_feats,
                                       heads=multi_head, concat=True))
        self.gc.append(GATv2Conv(in_channels=dims[0] * multi_head, out_channels=dims[1], edge_dim=num_edge_feats,
                                   heads=multi_head, concat=True))
        self.bn_gc = nn.BatchNorm1d(dims[0] * multi_head)

        # fully-connected layers for molecular features
        self.fc_m = nn.ModuleList()
        self.fc_m.append(nn.Linear(n_mol_feats, dims[0]))
        self.bn_m = nn.BatchNorm1d(dims[0])
        if n_layers[1] > 2:
            for i in range(0, n_layers[1] - 2):
                self.fc_m.append(nn.Linear(dims[0], dims[0]))
        self.fc_m.append(nn.Linear(dims[0], dims[1]))

        # fully-connected layers for all features
        self.fc = nn.ModuleList()
        self.fc.append(nn.Linear(dims[1] * multi_head + dims[1], dims[2]))
        for i in range(0, n_layers[2] - 3):
            self.fc.append(nn.Linear(dims[2], dims[2]))
        self.fc.append(nn.Linear(dims[2], dims[3]))
        self.fc.append(nn.Linear(dims[3], dims[-1]))

        self.gc_afp = AttentiveFP(in_channels=num_node_feats, hidden_channels=dims[0], out_channels=dims[-1], edge_dim=num_edge_feats,
                    num_timesteps=3, num_layers=n_layers[0], dropout=0.1)

    def forward(self, g):
        h = F.silu(self.bn_gc(self.gc[0](g.x, g.edge_index)))
        for i in range(1, self.n_layers[0]):
            h = F.silu(self.gc[i](h, g.edge_index))
        hg = global_add_pool(h, g.batch)

        h_m = F.silu(self.bn_m(self.fc_m[0](g.mol_feats)))
        for i in range(1, self.n_layers[1]):
            h_m = F.silu(self.fc_m[i](h_m))

        hg = torch.cat([hg, h_m], dim=1)
        for i in range(0, self.n_layers[2] - 1):
            hg = F.silu(self.fc[i](hg))
        out = self.fc[-1](hg) + self.gc_afp(g.x, g.edge_index, g.edge_attr, g.batch)

        return out

    def prt_emb(self, g):
        h = F.silu(self.bn_gc(self.gc[0](g.x, g.edge_index)))
        for i in range(1, self.n_layers[0]):
            h = F.silu(self.gc[i](h, g.edge_index))
        hg = global_add_pool(h, g.batch)

        h_m = F.silu(self.bn_m(self.fc_m[0](g.mol_feats)))
        for i in range(1, self.n_layers[1]):
            h_m = F.silu(self.fc_m[i](h_m))

        hg = torch.cat([hg, h_m], dim=1)
        for i in range(0, self.n_layers[2] - 1):
            hg = F.silu(self.fc[i](hg))

        return hg
