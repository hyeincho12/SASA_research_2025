import numpy
import random
import pandas
import torch
import xgboost as xgb
import utils.mol_dml
import utils.chem as chem
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score
from utils.models import GCN, GATv2, AttFP, GATv2AFP
from utils.models_beta import EGCN, GIN, MPNN, TFNN, GAT


# Experiment settings
gnn = 'tfnn'
testcode = '1'
dataset_name = 'zinc_2000'
batch_size = 32
init_lr = 1e-4
l2_coeff = 1e-6
n_epochs = 100
rand_seed = 10
n_models = 1
patience = 100
dims = [512, 512, 256, 256, 128]
n_layers = [3, 3, 3]
n_fp = 128
n_radius = 4
n_mol_feats = n_fp + 188
multi_head = 3
task = 'reg' # clf or 'reg'



# Load dataset
print('Load molecular structures...')
data, list_atom_types = chem.load_dataset('data/' + dataset_name + '.xlsx', n_fp, n_radius, task)
random.shuffle(data)
smiles = [x[0] for x in data]
mols = [x[1] for x in data]


# Generate training and test datasets
n_train = int(0.8 * len(data))
train_data = mols[:n_train]
test_data = mols[n_train:]
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
emb_loader = DataLoader(train_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

train_smiles = numpy.array(smiles[:n_train]).reshape(-1, 1)
test_smiles = numpy.array(smiles[n_train:]).reshape(-1, 1)

train_targets = numpy.array([x.y.item() for x in train_data]).reshape(-1, 1)
test_targets = numpy.array([x.y.item() for x in test_data]).reshape(-1, 1)


# Model configuration
if gnn == 'gcn':
    emb_net = GCN(chem.n_atom_feats, n_mol_feats, dims, n_layers)
elif gnn == 'attfp':
    emb_net = AttFP(chem.n_atom_feats, chem.n_bond_feats, n_mol_feats, dims, n_layers)
elif gnn == 'gatv2':
    emb_net = GATv2(chem.n_atom_feats, chem.n_bond_feats, n_mol_feats, multi_head, dims, n_layers)
elif gnn == 'gatv2afp':
    emb_net = GATv2AFP(chem.n_atom_feats, chem.n_bond_feats, n_mol_feats, multi_head, dims, n_layers)
elif gnn == 'tfnn':
    emb_net = TFNN(chem.n_atom_feats, chem.n_bond_feats, n_mol_feats, multi_head, dims, n_layers)
elif gnn == 'egcn':
    emb_net = EGCN(chem.n_atom_feats, n_mol_feats, dims, n_layers)
elif gnn == 'gin':
    emb_net = GIN(chem.n_atom_feats, n_mol_feats, dims, n_layers)
elif gnn == 'mpnn':
    emb_net = MPNN(chem.n_atom_feats, chem.n_bond_feats, n_mol_feats, multi_head, dims, n_layers)
else:
    print('Model {} is not available!'.format(gnn))
    exit()

optimizer = torch.optim.Adam(emb_net.parameters(), lr=init_lr, weight_decay=l2_coeff)


# Train GNN-based embedding network
print('Train the GNN-based embedding network...')
for i in range(0, n_epochs):
    train_loss = utils.mol_dml.train(emb_net, optimizer, train_loader)
    print('Epoch [{}/{}]\tTrain loss: {:.4f}'.format(i + 1, n_epochs, train_loss))


# Generate embeddings of the molecules
train_embs = utils.mol_dml.test(emb_net, emb_loader)
test_embs = utils.mol_dml.test(emb_net, test_loader)
train_emb_results = numpy.concatenate([train_embs, train_smiles, train_targets], axis=1).tolist()
test_emb_results = numpy.concatenate([test_embs, test_smiles, test_targets], axis=1).tolist()
df = pandas.DataFrame(train_emb_results)
df.to_excel('preds/preds_' + dataset_name + '_' +gnn + '_' + testcode +'_train.xlsx', header=None, index=None)
df = pandas.DataFrame(test_emb_results)
df.to_excel('preds/preds_' + dataset_name + '_' +gnn + '_' + testcode  +'_test.xlsx', header=None, index=None)


# Train XGBoost using the molecular embeddings
print('Train the XGBoost regressor...')
model = xgb.XGBRegressor(max_depth=8, n_estimators=300, subsample=0.8)
model.fit(train_embs, train_targets)
preds = model.predict(test_embs).reshape(-1, 1)
test_mae = numpy.mean(numpy.abs(test_targets - preds))
r2 = r2_score(test_targets, preds)
print('Test MAE: {:.4f}\tTest R2 score: {:.4f}'.format(test_mae, r2))


# Save prediction results
pred_results = list()
for i in range(0, preds.shape[0]):
    pred_results.append([test_smiles[i], test_targets[i].item(), preds[i].item()])
df = pandas.DataFrame(pred_results)
df.columns = ['smiles', 'true_y', 'pred_y']
df.to_excel('preds/preds_' + dataset_name + '_' +gnn + '_' + testcode + '_dml.xlsx', index=False)
