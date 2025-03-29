import numpy
import pandas
import torch
from tqdm import tqdm
from mendeleev.fetch import fetch_table
from sklearn import preprocessing
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import MACCSkeys

elem_feat_names = ['atomic_weight', 'atomic_radius', 'dipole_polarizability', 'vdw_radius', 'covalent_radius_bragg',
                   'en_pauling']

n_atom_feats = len(elem_feat_names) + 9
n_bond_feats = 6


def get_elem_feats():
    tb_atom_feats = fetch_table('elements')
    elem_feats = numpy.nan_to_num(numpy.array(tb_atom_feats[elem_feat_names]))

    return preprocessing.scale(elem_feats)


def load_dataset(path_user_dataset, n_fp, n_radius, task):
    elem_feats = get_elem_feats()
    list_mols = list()
    id_target = numpy.array(pandas.read_excel(path_user_dataset))
    list_atom_types = list()

    for i in tqdm(range(0, id_target.shape[0])):

        mol = smiles_to_mol_graph(elem_feats, n_fp, n_radius, id_target[i, 0], idx=i, target=id_target[i, 1],
                                  list_atom_types=list_atom_types, task=task)

        if mol is not None:
            list_mols.append((id_target[i, 0], mol, id_target[i, 1]))

    return list_mols, list_atom_types


def smiles_to_mol_graph(elem_feats, n_fp, n_radius, smiles, idx, target, list_atom_types, task):
    # smiles to a RDKit object
    # mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    mol = Chem.MolFromSmiles(smiles)

    # initialize lists
    atom_feats = list()
    atom_types = list()
    bonds = list()
    list_nbrs = list()
    bond_feats = list()

    # number of atoms
    n_atoms = mol.GetNumAtoms()

    # adjacency matrix
    adj_mat = Chem.GetAdjacencyMatrix(mol)
    adj_mat = adj_mat + numpy.eye(adj_mat.shape[0], dtype=int)

    # find atoms in ring structures
    atom_in_ring = numpy.array([mol.GetAtomWithIdx(i).IsInRing() for i in range(mol.GetNumAtoms())], dtype=int)
    # find atoms in aromatic rings
    atom_aroma = numpy.array([mol.GetAtomWithIdx(i).GetIsAromatic() for i in range(mol.GetNumAtoms())], dtype=int)
    # formal charge
    formal_q = numpy.array([mol.GetAtomWithIdx(i).GetFormalCharge() for i in range(mol.GetNumAtoms())], dtype=int)
    # atomic charge
    AllChem.ComputeGasteigerCharges(mol)
    atom_q = numpy.array([mol.GetAtomWithIdx(i).GetDoubleProp('_GasteigerCharge') for i in range(mol.GetNumAtoms())])

    # atomistic contributions to partition coefficient
    logp_contrib = numpy.array(rdMolDescriptors._CalcCrippenContribs(mol))

    # atomic features
    for i in range(0, n_atoms):
        atom = mol.GetAtoms()[i]
        atom_feats_rdkit = numpy.array([atom.GetExplicitValence(), atom.GetNumImplicitHs(),
                                        atom.GetNumExplicitHs(), atom_in_ring[i], atom_aroma[i],
                                        formal_q[i], logp_contrib[i][0], logp_contrib[i][1], atom_q[i]])
        tmp_feats = numpy.append(elem_feats[atom.GetAtomicNum() - 1, :], atom_feats_rdkit)
        atom_feats.append(tmp_feats)

        # find the atom type matched in list
        found = 0
        for j in range(0, len(list_atom_types)):
            if list_atom_types[j] == [atom.GetAtomicNum(), atom.GetHybridization()]:
                atom_types.append(j)
                found = 1
                break
        # add new atom type to list
        if found == 0:
            atom_types.append(len(list_atom_types))
            list_atom_types.append([atom.GetAtomicNum(), atom.GetHybridization()])

    # bond feature
    for i in range(0, n_atoms):
        list_nbrs.append(list())
        list_nbrs[i].append(i)
        bonds.append([i, i])
        bond_feats.append(numpy.zeros(n_bond_feats))

        for j in range(0, n_atoms):
            if adj_mat[i, j] == 1:
                list_nbrs[i].append(j)
                if i != j:
                    bonds.append([i, j])

                    bond = mol.GetBondBetweenAtoms(i, j)
                    b_type = bond.GetBondType()

                    tmp_feats = numpy.array([
                        b_type == Chem.rdchem.BondType.SINGLE,
                        b_type == Chem.rdchem.BondType.DOUBLE,
                        b_type == Chem.rdchem.BondType.TRIPLE,
                        b_type == Chem.rdchem.BondType.AROMATIC,
                        (bond.GetIsConjugated() if b_type is not None else 0),
                        (bond.IsInRing() if b_type is not None else 0)
                    ], dtype=float)
                else:
                    bonds.append([i, j])
                    tmp_feats = numpy.ones(n_bond_feats)

                bond_feats.append(tmp_feats)

    if len(bonds) == 0:
        return None

    # list -> numpy array -> pytorch tensor
    atom_feats = torch.tensor(numpy.array(atom_feats), dtype=torch.float)
    bonds = torch.tensor(bonds, dtype=torch.long).t().contiguous()

    bond_feats = torch.tensor(numpy.array(bond_feats), dtype=torch.float)
    if task == 'reg':
        y = torch.tensor(target, dtype=torch.float).view(-1, 1)
    elif task == 'clf':
        y = torch.tensor(target, dtype=torch.long).view(1)
    else:
        print('task {} is not available'.format(task))
        exit()
    atom_types = torch.tensor(numpy.array(atom_types), dtype=torch.int)

    mol_feats = list()
    mol_feats.append(ExactMolWt(mol))
    mol_feats.append(mol.GetRingInfo().NumRings())
    mol_feats.append(Descriptors.RingCount(mol))
    mol_feats.append(Descriptors.MolMR(mol))
    mol_feats.append(Descriptors.NumHAcceptors(mol))
    mol_feats.append(Descriptors.NumHDonors(mol))
    mol_feats.append(Descriptors.NumHeteroatoms(mol))
    mol_feats.append(Descriptors.NumRotatableBonds(mol))
    mol_feats.append(Descriptors.TPSA(mol))
    mol_feats.append(Descriptors.qed(mol))

    mol_feats.append(rdMolDescriptors.CalcLabuteASA(mol))
    mol_feats.append(rdMolDescriptors.CalcNumAliphaticHeterocycles(mol))
    mol_feats.append(rdMolDescriptors.CalcNumAliphaticRings(mol))
    mol_feats.append(rdMolDescriptors.CalcNumAmideBonds(mol))
    mol_feats.append(rdMolDescriptors.CalcNumAromaticCarbocycles(mol))
    mol_feats.append(rdMolDescriptors.CalcNumAromaticHeterocycles(mol))
    mol_feats.append(rdMolDescriptors.CalcNumAromaticRings(mol))
    mol_feats.append(rdMolDescriptors.CalcNumHeterocycles(mol))
    mol_feats.append(rdMolDescriptors.CalcNumSaturatedCarbocycles(mol))
    mol_feats.append(rdMolDescriptors.CalcNumSaturatedHeterocycles(mol))

    mol_feats.append(rdMolDescriptors.CalcNumSaturatedRings(mol))

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=n_radius, nBits=n_fp)
    morgan_array = numpy.zeros((0,), dtype=numpy.int8)
    Chem.DataStructs.ConvertToNumpyArray(fp, morgan_array)

    mol_feats = numpy.append(mol_feats, morgan_array)
    mol_feats = numpy.append(mol_feats, MACCSkeys.GenMACCSKeys(mol))
    mol_feats = torch.tensor(numpy.array(mol_feats), dtype=torch.float).view(1, n_fp + 188)

    return Data(x=atom_feats, y=y, edge_index=bonds, edge_attr=bond_feats, idx=idx, mol_feats=mol_feats,
                atom_types=atom_types, n_atoms=n_atoms)
