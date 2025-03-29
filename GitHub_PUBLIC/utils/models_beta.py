import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import GINConv
from torch_geometric.nn import NNConv
from torch_geometric.nn import TransformerConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import global_add_pool
from torch_geometric.data import Data

class GAT(nn.Module):
    # graph attention network (GAT)
    def __init__(self, num_node_feats, n_mol_feats, dims, n_layers):
        super(GAT, self).__init__()

        self.n_layers = n_layers

        # graph convolution layers
        self.gc = nn.ModuleList()
        self.gc.append(GATConv(num_node_feats, dims[0]))
        self.bn_gc = nn.BatchNorm1d(dims[0])
        if n_layers[0] > 2:
            for i in range(0, n_layers[0] - 2):
                self.gc.append(GATConv(dims[0], dims[0]))
        self.gc.append(GATConv(dims[0], dims[1]))

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



# Gyoung S. Na, Hyun Woo Kim, and Hyunju Chang,
# Costless Performance Improvement in Machine Learning for Graph-Based Molecular Analysis,
# J. Chem. Inf. Model, 2020, 60, 3, 1137-1145
class EGCN(nn.Module):
    # extended graph convolutional network
    def __init__(self, num_node_feats, n_mol_feats, dims, n_layers):
        super(EGCN, self).__init__()

        self.n_layers = n_layers

        # graph convolution layers
        self.gc = nn.ModuleList()
        self.gc.append(GCNConv(num_node_feats, dims[0]))
        self.bn_gc = nn.BatchNorm1d(dims[0])
        #배치 정규화 층


        if n_layers[0] > 2: #n_layer[0]-2개의 레이어
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
        hg = global_mean_pool(h, g.batch)

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
        hg = global_mean_pool(h, g.batch)

        h_m = F.silu(self.bn_m(self.fc_m[0](g.mol_feats)))
        for i in range(1, self.n_layers[1]):
            h_m = F.silu(self.fc_m[i](h_m))

        hg = torch.cat([hg, h_m], dim=1)
        for i in range(0, self.n_layers[2] - 1):
            hg = F.silu(self.fc[i](hg))

        return hg


# Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka,
# How Powerful are Graph Neural Networks?,
# International Conference on Learning Representations (ICLR) 2019
class GIN(nn.Module):
    # graph isomorphism network (GIN)
    def __init__(self, num_node_feats, n_mol_feats, dims, n_layers):
        super(GIN, self).__init__()
        self.n_layers = n_layers

        # graph convolution layers
        self.gc = nn.ModuleList()
        self.gc.append(GINConv(nn.Linear(num_node_feats, dims[0])))
        self.bn_gc = nn.BatchNorm1d(dims[0])
        if n_layers[0] > 2:
            for i in range(0, n_layers[0] - 2):
                self.gc.append(GINConv(nn.Linear(dims[0], dims[0])))
        self.gc.append(GINConv(nn.Linear(dims[0], dims[1])))

        self.fc_m = nn.ModuleList()
        self.fc_m.append(nn.Linear(n_mol_feats, dims[0]))
        self.bn_m = nn.BatchNorm1d(dims[0])
        if n_layers[1] > 2:
            for i in range(0, n_layers[1] - 2):
                self.fc_m.append(nn.Linear(dims[0], dims[0]))
        self.fc_m.append(nn.Linear(dims[0], dims[1]))

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



# Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, George E. Dahl
# Neural Message Passing for Quantum Chemistry
# International conference on machine learning. PMLR, 2017
class MPNN(nn.Module):
    def __init__(self, num_node_feats, num_edge_feats, n_mol_feats, dims, n_layers):
        super(MPNN, self).__init__()
        self.n_layers = n_layers

        self.gc = nn.ModuleList()

        self.nn1 = nn.Sequential(nn.Linear(num_edge_feats, 32), nn.ReLU(), nn.Linear(32, num_node_feats * dims[0]))
        self.nn2 = nn.Sequential(nn.Linear(num_edge_feats, 32), nn.ReLU(), nn.Linear(32, dims[0] * dims[0]))

        self.gc.append(NNConv(num_node_feats, dims[0], self.nn1))
        self.bn_gc=nn.BatchNorm1d(dims[0])
        if n_layers[0] > 2:
            for i in range(0, n_layers[0] - 2):
                self.gc.append(NNConv(dims[0], dims[0], self.nn2))
        self.gc.append(NNConv(dims[0], dims[1], self.nn2))

        self.fc_m = nn.ModuleList()
        self.fc_m.append(nn.Linear(n_mol_feats, dims[0]))
        self.bn_m = nn.BatchNorm1d(dims[0])
        if n_layers[1] > 2:
            for i in range(0,n_layers[1] - 2):
                self.fc_m.append(nn.Linear(dims[0], dims[0]))
        self.fc_m.append(nn.Linear(dims[0], dims[1]))

        self.fc = nn.ModuleList()
        self.fc.append(nn.Linear(dims[1] * 2, dims[2]))
        for i in range(0, n_layers[2]-3):
            self.fc.append(nn.Linear(dims[2], dims[2]))
        self.fc.append(nn.Linear(dims[2], dims[3]))
        self.fc.append(nn.Linear(dims[3], dims[-1]))

    def forward(self, g):
        # Apply graph convolution layers (NNConv) to graph node features
        h = F.silu(self.bn_gc(self.gc[0](g.x, g.edge_index, g.edge_attr)))
        for i in range(1, self.n_layers[0]):
            h = F.silu(self.gc[i](h, g.edge_index, g.edge_attr))
        hg = global_add_pool(h, g.batch)

        # Apply fully connected layers to molecular features
        h_m = F.silu(self.bn_m(self.fc_m[0](g.mol_feats)))
        for i in range(1, self.n_layers[1]):
            h_m = F.silu(self.fc_m[i](h_m))

        # Concatenate graph and molecular features
        hg = torch.cat([hg, h_m], dim=1)

        # Apply final fully connected layers
        for i in range(0, self.n_layers[2] - 1):
            hg = F.silu(self.fc[i](hg))
        out = self.fc[-1](hg)

        return out

    def prt_emb(self, g):
        # Apply NNConv layers to graph node features for embeddings
        h = F.silu(self.bn_gc(self.gc[0](g.x, g.edge_index, g.edge_attr)))
        for i in range(1, self.n_layers[0]):
            h = F.silu(self.gc[i](h, g.edge_index, g.edge_attr))
        hg = global_add_pool(h, g.batch)

        # Apply fully connected layers to molecular features for embeddings
        h_m = F.silu(self.bn_m(self.fc_m[0](g.mol_feats)))
        for i in range(1, self.n_layers[1]):
            h_m = F.silu(self.fc_m[i](h_m))

        # Concatenate graph and molecular feature embeddings
        hg = torch.cat([hg, h_m], dim=1)
        for i in range(0, self.n_layers[2] - 1):
            hg = F.silu(self.fc[i](hg))

        return hg


# Yunsheng Shi, Zhengjie Huang, Shikun Feng, Hui Zhong, Wenjin Wang, Yu Sun
#"The graph transformer operator from the `"Masked Label Prediction: Unified Message Passing Model for Semi-Supervised Classification"
#<https://arxiv.org/abs/2009.03509>, IJCAI 2021
class TFNN(nn.Module):
    def __init__(self, num_node_feats, num_edge_feats, n_mol_feats, multi_head ,dims, n_layers):
        super(TFNN, self).__init__()
        self.n_layers = n_layers

        # Graph Transformer convolution layers
        self.gc = nn.ModuleList()
        self.gc.append(TransformerConv(in_channels=num_node_feats, out_channels=dims[0],
                                       edge_dim=num_edge_feats, heads=multi_head, concat=True))
        self.bn_gc = nn.BatchNorm1d(dims[0] * multi_head)
        if n_layers[0] > 2:
            for i in range(0, n_layers[0] - 2):
                self.gc.append(TransformerConv(in_channels=dims[0] * multi_head, out_channels=dims[0],
                                               edge_dim=num_edge_feats, heads=multi_head, concat=True))
        self.gc.append(TransformerConv(in_channels=dims[0] * multi_head, out_channels=dims[1],
                                       edge_dim=num_edge_feats, heads=multi_head, concat=True))
        # Fully connected layers for molecular features
        self.fc_m = nn.ModuleList()
        self.fc_m.append(nn.Linear(n_mol_feats, dims[0]))
        self.bn_m = nn.BatchNorm1d(dims[0])
        if n_layers[1] > 2:
            for i in range(0, n_layers[1] - 2):
                self.fc_m.append(nn.Linear(dims[0], dims[0]))
        self.fc_m.append(nn.Linear(dims[0], dims[1]))

        # Fully connected layers for final output
        self.fc = nn.ModuleList()
        self.fc.append(nn.Linear(dims[1] * (multi_head + 1), dims[2]))
        for i in range(0, n_layers[2] - 3):
            self.fc.append(nn.Linear(dims[2], dims[2]))
        self.fc.append(nn.Linear(dims[2], dims[3]))
        self.fc.append(nn.Linear(dims[3], dims[-1]))

    def forward(self, g):
        # Apply graph convolution layers (NNConv) to graph node features
        h = F.silu(self.bn_gc(self.gc[0](g.x, g.edge_index, g.edge_attr)))
        for i in range(1, self.n_layers[0]):
             h = F.silu(self.gc[i](h, g.edge_index, g.edge_attr))
        hg = global_add_pool(h, g.batch)

        # Apply fully connected layers to molecular features
        h_m = F.silu(self.bn_m(self.fc_m[0](g.mol_feats)))
        for i in range(1, self.n_layers[1]):
            h_m = F.silu(self.fc_m[i](h_m))

        # Concatenate graph and molecular features
        hg = torch.cat([hg, h_m], dim=1)

        # Apply final fully connected layers
        for i in range(0, self.n_layers[2] - 1):
            hg = F.silu(self.fc[i](hg))
        out = self.fc[-1](hg)

        return out

    def prt_emb(self, g):
       # Apply NNConv layers to graph node features for embeddings
        h = F.silu(self.bn_gc(self.gc[0](g.x, g.edge_index, g.edge_attr)))
        for i in range(1, self.n_layers[0]):
            h = F.silu(self.gc[i](h, g.edge_index, g.edge_attr))
        hg = global_add_pool(h, g.batch)

        # Apply fully connected layers to molecular features for embeddings
        h_m = F.silu(self.bn_m(self.fc_m[0](g.mol_feats)))
        for i in range(1, self.n_layers[1]):
            h_m = F.silu(self.fc_m[i](h_m))

        # Concatenate graph and molecular feature embeddings
        hg = torch.cat([hg, h_m], dim=1)
        for i in range(0, self.n_layers[2] - 1):
            hg = F.silu(self.fc[i](hg))

        return hg

