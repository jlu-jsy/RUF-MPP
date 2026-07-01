# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Molecular utility functions for scaffold extraction and penalized logP."""

from rdkit import Chem
from rdkit.Chem import RDConfig
from rdkit.Chem.Scaffolds import MurckoScaffold

def atom_valences(atom_types):
    """Return the maximum valence for each atom type.

    This is not a count of valence electrons, but the maximum number of bonds
    each element can form. For example, ["C", "H", "O"] returns [4, 1, 2].

    Args:
        atom_types: List of atom symbols, e.g., ["C", "H", "O"].

    Returns:
        List of integer atom valences.
    """
    periodic_table = Chem.GetPeriodicTable()
    return [
        max(list(periodic_table.GetValenceList(atom_type)))
        for atom_type in atom_types
    ]


def get_scaffold(mol):
    """Compute the Bemis-Murcko scaffold of an RDKit molecule.

    Args:
        mol: RDKit Mol object.

    Returns:
        Scaffold SMILES string.
    """
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold, isomericSmiles=True)