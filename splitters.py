import random
from collections import defaultdict

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm


def generate_scaffold(mol, include_chirality=False):
    """Generate the Bemis-Murcko scaffold SMILES for a molecule.

    Args:
        mol: RDKit Mol object or SMILES string.
        include_chirality: Whether to include chirality information in the scaffold.

    Returns:
        Scaffold SMILES string, or None if scaffold generation fails.
    """
    try:
        mol = Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol
        if mol is None:
            return None
        return MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol,
            includeChirality=include_chirality,
        )
    except Exception as exc:
        print(f"[Scaffold Error] Failed to generate scaffold for molecule: {mol}. Error: {exc}")
        return None


def scaffold_to_smiles(mols, use_indices=False):
    """Map each scaffold to either molecule SMILES strings or molecule indices."""
    scaffolds = defaultdict(set)

    for idx, mol in tqdm(enumerate(mols), total=len(mols)):
        scaffold = generate_scaffold(mol)
        if scaffold is None:
            continue

        if use_indices:
            scaffolds[scaffold].add(idx)
        else:
            scaffolds[scaffold].add(mol)

    return scaffolds


def scaffold_split(
    dataset,
    smiles_list,
    balanced=False,
    frac_train=0.8,
    frac_valid=0.1,
    frac_test=0.1,
    seed=0,
):
    """Split a molecular dataset by Bemis-Murcko scaffold.

    Molecules sharing the same scaffold are assigned to the same split to reduce
    scaffold leakage between training, validation, and test sets.
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    train_size = frac_train * len(dataset)
    valid_size = frac_valid * len(dataset)
    test_size = frac_test * len(dataset)

    train_idx, valid_idx, test_idx = [], [], []

    scaffold_to_indices = scaffold_to_smiles(smiles_list, use_indices=True)
    index_sets = list(scaffold_to_indices.values())

    if balanced:
        large_index_sets = []
        small_index_sets = []

        for index_set in index_sets:
            if len(index_set) > valid_size / 2 or len(index_set) > test_size / 2:
                large_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)

        random.seed(seed)
        random.shuffle(large_index_sets)
        random.shuffle(small_index_sets)
        index_sets = large_index_sets + small_index_sets
    else:
        index_sets = sorted(index_sets, key=len, reverse=True)

    for index_set in index_sets:
        if len(train_idx) + len(index_set) <= train_size:
            train_idx.extend(index_set)
        elif len(valid_idx) + len(index_set) <= valid_size:
            valid_idx.extend(index_set)
        else:
            test_idx.extend(index_set)

    train_set = set(train_idx)
    valid_set = set(valid_idx)
    test_set = set(test_idx)

    assert train_set.isdisjoint(valid_set)
    assert train_set.isdisjoint(test_set)
    assert valid_set.isdisjoint(test_set)

    train_dataset = dataset[torch.tensor(train_idx, dtype=torch.long)]
    valid_dataset = dataset[torch.tensor(valid_idx, dtype=torch.long)]
    test_dataset = dataset[torch.tensor(test_idx, dtype=torch.long)]

    return train_dataset, valid_dataset, test_dataset


def random_split(
    dataset,
    frac_train=0.8,
    frac_valid=0.1,
    frac_test=0.1,
    seed=0,
):
    """Randomly split a dataset into training, validation, and test subsets."""
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    if seed is not None:
        random.seed(seed)

    indices = list(range(len(dataset)))
    random.shuffle(indices)

    train_size = int(frac_train * len(dataset))
    valid_size = int(frac_valid * len(dataset))

    train_dataset = dataset[indices[:train_size]]
    valid_dataset = dataset[indices[train_size:train_size + valid_size]]
    test_dataset = dataset[indices[train_size + valid_size:]]

    return train_dataset, valid_dataset, test_dataset
