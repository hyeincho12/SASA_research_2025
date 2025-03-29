import os
os.environ['OMP_NUM_THREADS'] = "2"

import numpy
import random
import pandas
import torch
import utils.ml
import utils.chem as chem
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score, accuracy_score, f1_score, roc_auc_score
from utils.models import GCN, GATv2, AttFP, GATv2AFP
from utils.models_beta import EGCN, GIN, MPNN, TFNN, GAT
torch.set_num_threads(2)

# Experiment settings
gnn = 'tfnn'
dataset_name = 'esol'
testcode = '1'
batch_size = 32
init_lr = 1e-4
l2_coeff = 1e-6
n_epochs = 100
rand_seed = 10
n_models = 1
patience = 100
dims = [32, 16, 16, 8, 1]
n_layers = [3, 3, 3]
n_fp = 128
n_radius = 4
n_mol_feats = n_fp + 188
multi_head = 3
task = 'reg' # clf or 'reg'

# Load dataset
print('Load molecular structures...')
data, list_atom_types = chem.load_dataset('data/' + dataset_name + '.xlsx', n_fp, n_radius, task)
n_atom_types = len(list_atom_types)

random.seed(rand_seed)
random.shuffle(data)
smiles = [x[0] for x in data]
mols = [x[1] for x in data]

# Generate training and test datasets
n_train = int(0.8 * len(data))
n_valid = int(0.1 * len(data))
n_test = len(data) - n_train - n_valid

train_data = mols[:n_train]
valid_data = mols[n_train:(n_train + n_valid)]
train_data_all = mols[:(n_train + n_valid)]
test_data = mols[(n_train + n_valid):]

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
train_loader_all = DataLoader(train_data_all, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

test_smiles = smiles[n_train:]

train_targets = numpy.array([x.y.item() for x in train_data]).reshape(-1, 1)
valid_targets = numpy.array([x.y.item() for x in valid_data]).reshape(-1, 1)
test_targets = numpy.array([x.y.item() for x in test_data]).reshape(-1, 1)

# Model configuration
if task == 'reg':
    criterion = torch.nn.MSELoss()

    valid_rmse = 0.0
    valid_mae = 0.0
    valid_r2 = 0.0

elif task == 'clf':
    criterion = torch.nn.CrossEntropyLoss()

    valid_acc = 0.0
    valid_f1 = 0.0
    valid_roc_auc = 0.0
else:
    print('task {} is not available'.format(task))
    exit()

# Train graph neural network (GNN)
print('Train the GNN-based predictor...')
if gnn == 'gcn':
    model_epoch_opt = GCN(chem.n_atom_feats, n_mol_feats, dims, n_layers)
    #num_node_feats, n_mol_feats, dims, n_layers
elif gnn == 'attfp':
    model_epoch_opt = AttFP(chem.n_atom_feats, chem.n_bond_feats, n_mol_feats, dims, n_layers)
    #num_node_feats, num_edge_feats, n_mol_feats, dims, n_layers
elif gnn == 'gatv2':
    model_epoch_opt = GATv2(chem.n_atom_feats, chem.n_bond_feats, n_mol_feats, multi_head, dims, n_layers)
elif gnn == 'gatv2afp':
    model_epoch_opt = GATv2AFP(chem.n_atom_feats, chem.n_bond_feats, n_mol_feats, multi_head, dims, n_layers)
elif gnn == 'egcn':
    model_epoch_opt = EGCN(chem.n_atom_feats, n_mol_feats, dims, n_layers)
elif gnn == 'gin':
    model_epoch_opt = GIN(chem.n_atom_feats, n_mol_feats, dims, n_layers)
elif gnn == 'mpnn':
    model_epoch_opt = MPNN(chem.n_atom_feats, chem.n_bond_feats, n_mol_feats, dims, n_layers)
elif gnn == 'tfnn':
    model_epoch_opt = TFNN(chem.n_atom_feats, chem.n_bond_feats,n_mol_feats, multi_head,dims, n_layers)
elif gnn == 'gat':
    model_epoch_opt = GAT(chem.n_atom_feats, n_mol_feats, dims, n_layers)
else:
    print('Model {} is not available!'.format(gnn))
    exit()

valid_err_list = list()
n_epoch_opt = -1
for i in range(0, n_epochs):
    optimizer = torch.optim.Adam(model_epoch_opt.parameters(), lr=init_lr, weight_decay=l2_coeff)
    train_loss = utils.ml.train(model_epoch_opt, optimizer, train_loader, criterion)

    valid_preds = utils.ml.test(model_epoch_opt, valid_loader)
    print('Epoch [{}/{}]\tTrain loss: {:.4f}'.format(i + 1, n_epochs, train_loss))

    if task == 'reg':
        valid_rmse_now = numpy.sqrt(numpy.mean((valid_targets - valid_preds) ** 2))

        if i > patience:
            if numpy.mean(valid_err_list[-patience:]) > numpy.mean(valid_err_list[-patience - 1:-2]):
                n_epoch_opt = i
                break

        valid_err_list.append(valid_rmse_now)
    elif task == 'clf':
        valid_preds_clf = numpy.argmax(valid_preds, axis=1)
        valid_acc_now = roc_auc_score(valid_targets, valid_preds_clf)

        if i > patience:
            if numpy.mean(valid_err_list[-patience:]) > numpy.mean(valid_err_list[-patience - 1:-2]):
                n_epoch_opt = i
                break

        valid_err_list.append(valid_acc_now)
    else:
        print('Error: task not found')
        exit()

if task == 'reg':
    valid_preds = utils.ml.test(model_epoch_opt, valid_loader)
    valid_rmse = numpy.sqrt(numpy.mean((valid_targets - valid_preds) ** 2))
    valid_mae = numpy.mean(numpy.abs(valid_targets - valid_preds))
    valid_r2 = r2_score(valid_targets, valid_preds)

    print('VALIDATION: MAE(gnn): {:.4f}\tRMSE(gnn): {:.4f}\tR2 score: {:.4f}'.format(valid_mae, valid_rmse, valid_r2))
elif task == 'clf':
    valid_preds = utils.ml.test(model_epoch_opt, valid_loader)
    valid_preds_clf = numpy.argmax(valid_preds, axis=1)
    valid_acc = accuracy_score(valid_targets, valid_preds_clf)
    valid_f1 = f1_score(valid_targets, valid_preds_clf)
    valid_roc_auc = roc_auc_score(valid_targets, valid_preds_clf)

    print('VALIDATION: ACC (gnn): {:.4f}\tF1 score: {:.4f}\tROC-AUC: {:.4f}'.format(valid_acc, valid_f1, valid_roc_auc))
else:
    print('Error: task not found')
    exit()

# Train GNN model(s) using all training data
models = list()
if gnn == 'gcn':
    for i in range(0, n_models):
        models.append(GCN(chem.n_atom_feats, n_mol_feats, dims, n_layers))
elif gnn == 'attfp':
    for i in range(0, n_models):
        models.append(AttFP(chem.n_atom_feats, chem.n_bond_feats, n_mol_feats, dims, n_layers))
elif gnn == 'gatv2':
    for i in range(0, n_models):
        models.append(GATv2(chem.n_atom_feats, chem.n_bond_feats, n_mol_feats, multi_head, dims, n_layers))
elif gnn == 'gatv2afp':
    for i in range(0, n_models):
        models.append(GATv2AFP(chem.n_atom_feats, chem.n_bond_feats, n_mol_feats, multi_head, dims, n_layers))
elif gnn == 'egcn':
    for i in range(0,n_models):
        models.append(EGCN(chem.n_atom_feats, n_mol_feats, dims, n_layers))
elif gnn == 'gin':
    for i in range(0, n_models):
        models.append(GIN(chem.n_atom_feats, n_mol_feats, dims, n_layers))
elif gnn == 'mpnn':
    for i in range(0, n_models):
        models.append(MPNN(chem.n_atom_feats, chem.n_bond_feats, n_mol_feats, dims, n_layers))
elif gnn == 'tfnn':
    for i in range(0, n_models):
        models.append(TFNN(chem.n_atom_feats, chem.n_bond_feats, n_mol_feats, multi_head,dims, n_layers))
elif gnn == 'gat':
    for i in range(0, n_models):
        models.append(GAT(chem.n_atom_feats, n_mol_feats, dims, n_layers))
else:
    print('Model {} is not available!'.format(gnn))
    exit()

for i in range(0, n_epoch_opt):
    train_loss = 0.0
    for j in range(0, n_models):
        optimizer = torch.optim.Adam(models[j].parameters(), lr=init_lr, weight_decay=l2_coeff)
        train_loss += utils.ml.train(models[j], optimizer, train_loader_all, criterion)

    print('Epoch [{}/{}]\tTrain loss: {:.4f}'.format(i + 1, n_epochs, train_loss / n_models))

# Test the trained GNN
if task == 'reg':
    preds = utils.ml.test(models[0], test_loader)
    for j in range(1, n_models):
        preds += utils.ml.test(models[j], test_loader)
    preds /= n_models

    test_mae = numpy.mean(numpy.abs(test_targets - preds))
    test_rmse = numpy.sqrt(numpy.mean((test_targets - preds) ** 2))
    r2 = r2_score(test_targets, preds)
    print('Test: MAE(gnn): {:.4f}\tTest RMSE(gnn): {:.4f}\tTest R2 score: {:.4f}'.format(test_mae, test_rmse, r2))
elif task == 'clf':
    preds_prob = utils.ml.test(models[0], test_loader)
    for j in range(1, n_models):
        preds_prob += utils.ml.test(models[j], test_loader)
    preds_prob /= n_models
    preds = numpy.argmax(preds_prob, axis=1)
    test_acc = accuracy_score(test_targets, preds)
    f1 = f1_score(test_targets, preds)
    test_roc_auc = roc_auc_score(test_targets, preds)
    print('Test: ACC (gnn): {:.4f}\tF1 score: {:.4f}\tROC-AUC: {:.4f}'.format(test_acc, f1, test_roc_auc))
else:
    print('Error: task not found')
    exit()

#save prediction results
pred_results = list()
for i in range(0, preds.shape[0]):
    pred_results.append([test_smiles[i], test_targets[i].item(), preds[i].item()])
df = pandas.DataFrame(pred_results)
df.columns = ['smiles', 'true_y', 'pred_y']
df.to_excel('preds/preds_' + dataset_name + '_' +gnn + '_' + testcode + '_gnn.xlsx', index=False)
