import numpy as np
import torch
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from torch_geometric.data import Data

def smile_list2graph_list(smiles_list, y):
    """Convert a list of SMILES strings and labels into PyG graph objects."""
    graph_list = []

    for idx, molecule_smiles in enumerate(smiles_list):
        graph = Data()
        graph.smiles = molecule_smiles

        molecule = Chem.MolFromSmiles(molecule_smiles)
        molecule = Chem.AddHs(molecule)

        atom_features = [atom_to_feature_vector(atom) for atom in molecule.GetAtoms()]
        x = np.asarray(atom_features, dtype=np.int64)

        num_bond_features = 3
        if molecule.GetNumBonds() > 0:
            edges = []
            edge_features = []

            for bond in molecule.GetBonds():
                begin_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()
                edge_feature = bond_to_feature_vector(bond)

                edges.append((begin_idx, end_idx))
                edge_features.append(edge_feature)
                edges.append((end_idx, begin_idx))
                edge_features.append(edge_feature)

            edge_index = np.asarray(edges, dtype=np.int64).T
            edge_attr = np.asarray(edge_features, dtype=np.int64)
        else:
            edge_index = np.empty((2, 0), dtype=np.int64)
            edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

        graph.edge_index = torch.tensor(edge_index, dtype=torch.long)
        graph.edge_attr = torch.from_numpy(edge_attr)
        graph.x = torch.from_numpy(x)
        graph.y = torch.from_numpy(np.asarray([np.asarray(y[idx])]))
        graph.batch = torch.zeros(x.shape[0]).long()

        graph_list.append(graph)

    return graph_list

def get_valid_actions(
    state,
    atom_types,
    allow_removal,
    allow_no_modification,
    allowed_ring_sizes,
    allow_bonds_between_rings,
):
    """Compute valid molecule-edit actions for the current SMILES state."""
    if not state:
        return copy.deepcopy(atom_types)

    mol = Chem.MolFromSmiles(state)
    if mol is None:
        raise ValueError(f"Received invalid state: {state}")

    atom_valences = {
        atom_type: molecules.atom_valences([atom_type])[0]
        for atom_type in atom_types
    }

    atoms_with_free_valence = {}
    for valence in range(1, max(atom_valences.values())):
        atoms_with_free_valence[valence] = [
            atom.GetIdx()
            for atom in mol.GetAtoms()
            if atom.GetNumImplicitHs() >= valence
        ]

    valid_actions = set()
    valid_actions.update(
        _atom_addition(
            mol,
            atom_types=atom_types,
            atom_valences=atom_valences,
            atoms_with_free_valence=atoms_with_free_valence,
        )
    )
    valid_actions.update(
        _bond_addition(
            mol,
            atoms_with_free_valence=atoms_with_free_valence,
            allowed_ring_sizes=allowed_ring_sizes,
            allow_bonds_between_rings=allow_bonds_between_rings,
        )
    )

    if allow_removal:
        valid_actions.update(_bond_removal(mol))
    if allow_no_modification:
        valid_actions.add(Chem.MolToSmiles(mol))

    return valid_actions