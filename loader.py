import json
import os
import pickle
import warnings
from copy import deepcopy
from itertools import repeat
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from cite.aug import drop_nodes, permute_edges, subgraph
from enviroment import smile_list2graph_list


RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


RULE_STRATEGIES = ("atom_replace", "bond_change", "scaffold_hop")
ATOM_REPLACEMENTS = ("C", "N", "O", "F", "Cl", "Br", "S")
SCAFFOLD_FRAGMENTS = ("C", "CC", "CCO", "CN", "CO", "CF")
SIMILARITY_RANGE = (0.20, 0.95)


def pick_non_aromatic_atom(mol: Chem.RWMol, max_tries: int = 10) -> Optional[int]:
    """Sample a non-aromatic atom index when possible."""
    for _ in range(max_tries):
        idx = np.random.randint(0, mol.GetNumAtoms())
        if not mol.GetAtomWithIdx(idx).GetIsAromatic():
            return idx
    return None


def calculate_similarity(smiles1: str, smiles2: str) -> float:
    """Return the average Tanimoto similarity from Morgan and RDKit fingerprints."""
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        if mol1 is None or mol2 is None:
            return 0.0

        fp1_morgan = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=1024)
        fp2_morgan = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=1024)
        sim_morgan = DataStructs.TanimotoSimilarity(fp1_morgan, fp2_morgan)

        fp1_rdkit = Chem.RDKFingerprint(mol1)
        fp2_rdkit = Chem.RDKFingerprint(mol2)
        sim_rdkit = DataStructs.TanimotoSimilarity(fp1_rdkit, fp2_rdkit)

        return float((sim_morgan + sim_rdkit) / 2.0)
    except Exception:
        return 0.0


def is_valid_graph(data: Optional[Data]) -> bool:
    """Check whether a PyG molecular graph contains valid node and edge tensors."""
    if data is None:
        return False
    if getattr(data, "num_nodes", 0) is None or getattr(data, "num_nodes", 0) == 0:
        return False
    if not hasattr(data, "x") or data.x is None or data.x.size(0) == 0:
        return False
    if torch.isnan(data.x).any():
        return False
    if hasattr(data, "edge_index"):
        edge_index = data.edge_index
        if edge_index is None or edge_index.numel() == 0:
            return False
        if edge_index.max().item() >= data.num_nodes or edge_index.min().item() < 0:
            return False
    return True


class RuleAugmentor:
    """RDKit-based molecular augmentation with validity and similarity filters."""

    def __init__(self, mode: str = "prefer_non_aromatic", probs: Sequence[float] = (0.45, 0.45, 0.10)):
        self.mode = mode
        self.probs = tuple(float(p) for p in probs)
        assert len(self.probs) == 3, "rule_probs must contain three values."
        assert abs(sum(self.probs) - 1.0) < 1e-6, "rule_probs must sum to 1.0."
        print(f"[Rule Aug] Initialized with mode={mode}, probs={self.probs}")

    def rule_augment_one(self, smiles: str) -> Optional[str]:
        """Generate one rule-augmented SMILES string, or return None on failure."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        strategy = np.random.choice(RULE_STRATEGIES, p=self.probs)

        try:
            if strategy == "atom_replace":
                new_mol = self._atom_replace(mol)
            elif strategy == "bond_change":
                new_mol = self._bond_change(mol)
            else:
                new_mol = self._scaffold_hop(mol)

            if new_mol is None:
                return None
            return self._sanitize_and_filter(smiles, new_mol)
        except Exception:
            return None

    def _atom_replace(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        rw_mol = Chem.RWMol(mol)

        if self.mode == "random":
            atom_idx = np.random.randint(0, rw_mol.GetNumAtoms())
        elif self.mode == "skip_aromatic":
            atom_idx = np.random.randint(0, rw_mol.GetNumAtoms())
            if rw_mol.GetAtomWithIdx(atom_idx).GetIsAromatic():
                return None
        else:
            atom_idx = pick_non_aromatic_atom(rw_mol)
            if atom_idx is None:
                return None

        old_symbol = rw_mol.GetAtomWithIdx(atom_idx).GetSymbol()
        candidates = [symbol for symbol in ATOM_REPLACEMENTS if symbol != old_symbol]
        new_symbol = np.random.choice(candidates)
        rw_mol.GetAtomWithIdx(atom_idx).SetAtomicNum(
            Chem.GetPeriodicTable().GetAtomicNumber(new_symbol)
        )
        return rw_mol.GetMol()

    @staticmethod
    def _bond_change(mol: Chem.Mol) -> Chem.Mol:
        rw_mol = Chem.RWMol(mol)
        if rw_mol.GetNumBonds() > 0:
            bond_idx = np.random.randint(0, rw_mol.GetNumBonds())
            bond = rw_mol.GetBondWithIdx(bond_idx)
            bond_type = np.random.choice([
                Chem.rdchem.BondType.SINGLE,
                Chem.rdchem.BondType.DOUBLE,
            ])
            bond.SetBondType(bond_type)
        return rw_mol.GetMol()

    @staticmethod
    def _scaffold_hop(mol: Chem.Mol) -> Optional[Chem.Mol]:
        rw_mol = Chem.RWMol(mol)
        if np.random.random() < 0.5 and rw_mol.GetNumAtoms() > 5:
            atom_idx = np.random.randint(0, rw_mol.GetNumAtoms())
            rw_mol.RemoveAtom(atom_idx)
            return rw_mol.GetMol()

        fragment = np.random.choice(SCAFFOLD_FRAGMENTS)
        fragment_mol = Chem.MolFromSmiles(fragment)
        if fragment_mol is None:
            return None
        return Chem.CombineMols(mol, fragment_mol)

    @staticmethod
    def _sanitize_and_filter(original_smiles: str, new_mol: Chem.Mol) -> Optional[str]:
        sanitize_status = Chem.SanitizeMol(new_mol, catchErrors=True)
        if sanitize_status != 0:
            return None

        new_smiles = Chem.MolToSmiles(new_mol)
        if new_smiles == original_smiles:
            return None
        if len(new_smiles) < 4:
            return None
        if new_mol.GetNumAtoms() < 3:
            return None
        if "." in new_smiles:
            return None

        similarity = calculate_similarity(original_smiles, new_smiles)
        low, high = SIMILARITY_RANGE
        if not (low <= similarity <= high):
            return None

        return new_smiles


class MoleculeDataset(InMemoryDataset):
    """Molecule dataset with optional rule-based or random graph augmentation."""

    def __init__(
        self,
        root: str,
        device,
        max_step: int = 3,
        transform=None,
        pre_transform=None,
        dataset_name: Optional[str] = None,
        aug: str = "none",
        aug_ratio: float = 0.0,
        rule_mode: str = "random",
        rule_probs: Sequence[float] = (0.45, 0.45, 0.10),
    ):
        self.dataset_name = dataset_name
        self.max_step = max_step
        self.aug = aug
        self.device = device
        self.aug_ratio = aug_ratio
        self.rule_mode = rule_mode
        self.rule_probs = self._parse_rule_probs(rule_probs)
        self.rule_augmentor = RuleAugmentor(mode=self.rule_mode, probs=self.rule_probs)

        super().__init__(root, transform, pre_transform)

        torch.serialization.add_safe_globals([Data])
        self.data, self.slices = torch.load(self.processed_paths[0])

        self._all_smiles = self._load_smiles_cache()
        self.atoms_ = self._load_atom_cache()

    @staticmethod
    def _parse_rule_probs(rule_probs: Sequence[float]) -> Tuple[float, float, float]:
        if isinstance(rule_probs, str):
            probs = tuple(float(x) for x in rule_probs.split(","))
        else:
            probs = tuple(float(x) for x in rule_probs)

        assert len(probs) == 3, "rule_probs must contain three values."
        assert abs(sum(probs) - 1.0) < 1e-6, "rule_probs must sum to 1.0."
        return probs

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        pass

    def _load_smiles_cache(self) -> Optional[list]:
        smiles_cache_path = os.path.join(self.processed_dir, "data_aug.pt")
        if os.path.exists(smiles_cache_path):
            smiles = torch.load(smiles_cache_path)
            print(f"[Dataset] Loaded cached SMILES list: {len(smiles)} items")
            return smiles
        print(f"[Dataset] No cached SMILES list found at {smiles_cache_path}")
        return None

    def _load_atom_cache(self) -> list:
        atom_path = os.path.join(self.processed_dir, "atom.json")
        if os.path.exists(atom_path):
            with open(atom_path, "r") as f:
                return json.load(f)
        return []

    def _load_raw(self):
        if hasattr(self, "_all_smiles") and hasattr(self, "_all_targets"):
            return self._all_smiles, self._all_targets

        df = pd.read_csv(f"data/{self.dataset_name}.csv")
        smiles_list = df["smiles"].astype(str).str.replace("\n", "", regex=False).tolist()
        targets = df.iloc[:, 1:].values.tolist()

        clean_smiles = []
        clean_targets = []
        for smiles, target in zip(smiles_list, targets):
            if Chem.MolFromSmiles(smiles) is None:
                continue
            clean_smiles.append(smiles)
            clean_targets.append(target)

        self._raw_smiles = clean_smiles
        self._raw_targets = np.array(clean_targets)
        return self._raw_smiles, self._raw_targets

    def get_atoms(self):
        atoms = []
        smiles_list, _ = self._load_raw()
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                if symbol not in atoms:
                    atoms.append(symbol)
        return atoms

    def smiles(self):
        if getattr(self, "_all_smiles", None) is not None:
            return self._all_smiles
        return self._load_raw()[0]

    def target(self):
        if getattr(self, "_all_targets", None) is not None:
            return self._all_targets
        return self._load_raw()[1]

    def process(self):
        print("[Dataset] Start processing SMILES to graphs...", flush=True)

        self.atoms_ = self.get_atoms()
        os.makedirs(self.processed_dir, exist_ok=True)
        with open(os.path.join(self.processed_dir, "atom.json"), "w") as f:
            json.dump(self.atoms_, f)

        smiles_list = self.smiles()
        targets = self.target()
        print(f"[Dataset] Raw molecules: {len(smiles_list)}, targets: {len(targets)}")

        data_list = []
        all_smiles = []
        all_targets = []
        kept_id = 0

        for raw_idx, (smiles, target) in tqdm(
            enumerate(zip(smiles_list, targets)),
            total=len(smiles_list),
            desc="Processing molecules",
        ):
            graph_list = smile_list2graph_list([smiles], [target])
            if len(graph_list) != 1:
                continue

            data = graph_list[0]
            if not is_valid_graph(data):
                continue

            data.idx = torch.tensor([kept_id])
            data.raw_idx = torch.tensor([raw_idx])
            data_list.append(data)
            all_smiles.append(smiles)
            all_targets.append(target)
            kept_id += 1

        print(
            f"[Dataset] Finished graph conversion: kept {len(data_list)} / {len(smiles_list)}",
            flush=True,
        )

        if self.aug == "rule" and self.aug_ratio > 0:
            augmented_data, augmented_smiles, augmented_targets = self._build_rule_augmented_data(
                smiles_list, targets
            )
            data_list.extend(augmented_data)
            all_smiles.extend(augmented_smiles)
            all_targets.extend(augmented_targets)
            print(
                f"[Rule Aug] Dataset expanded: {len(smiles_list)} -> {len(data_list)} "
                f"(+{len(augmented_data)})"
            )

        self._all_smiles = all_smiles
        self._all_targets = np.array(all_targets)

        smiles_cache_path = os.path.join(self.processed_dir, "data_aug.pt")
        torch.save(self._all_smiles, smiles_cache_path)
        print(f"[Dataset] Saved SMILES cache: {len(self._all_smiles)} items")

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _build_rule_augmented_data(self, smiles_list: Sequence[str], targets: Sequence[Sequence[float]]):
        print("[Rule Aug] Starting rule-based data augmentation...")

        n_new = int(len(smiles_list) * self.aug_ratio)
        if n_new <= 0:
            return [], [], []

        new_smiles = []
        new_targets = []
        attempts = 0
        sample_idx = np.random.choice(len(smiles_list), size=n_new * 5, replace=True)

        for idx in tqdm(sample_idx, desc="[Rule Aug] Generating molecules"):
            smiles = smiles_list[idx]
            target = targets[idx]
            attempts += 1

            candidate = self.rule_augmentor.rule_augment_one(smiles)
            if candidate is not None and candidate not in new_smiles:
                new_smiles.append(candidate)
                new_targets.append(target)

            if len(new_smiles) >= n_new:
                break

        success_rate = len(new_smiles) / attempts if attempts > 0 else 0.0
        print(f"[Rule Aug] Generated {len(new_smiles)} new molecules")
        print(f"[Rule Aug] attempts={attempts}, success_rate={success_rate:.3f}")

        new_data_list = smile_list2graph_list(new_smiles, new_targets)
        clean_data = []
        clean_smiles = []
        clean_targets = []

        for smiles, target, data in zip(new_smiles, new_targets, new_data_list):
            if not is_valid_graph(data):
                continue
            data.idx = torch.tensor([-1])
            data.raw_idx = torch.tensor([-1])
            clean_data.append(data)
            clean_smiles.append(smiles)
            clean_targets.append(target)

        print(f"[Rule Aug] Valid graphs kept: {len(clean_data)} / {len(new_data_list)}")
        return clean_data, clean_smiles, clean_targets

    def get(self, idx):
        data = self.data.__class__()

        if hasattr(self.data, "__num_nodes__"):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.keys():
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                index = list(repeat(slice(None), item.dim()))
                cat_dim = self.data.__cat_dim__(key, item)
                index[cat_dim] = slice(slices[idx], slices[idx + 1])
            else:
                index = slice(slices[idx], slices[idx + 1])
            data[key] = item[index]

        if not hasattr(data, "idx"):
            data.idx = torch.tensor([idx])

        if self.aug == "random":
            data = self._get_random_augmented_graph(data, idx)

        return data.cpu()

    def _get_random_augmented_graph(self, data: Data, idx: int) -> Data:
        graph_path = f"random/{self.dataset_name}/pyg/graph_{idx}_1"
        if os.path.exists(graph_path):
            with open(graph_path, "rb") as f:
                return pickle.load(f)

        aug_type = np.random.randint(3)
        if aug_type == 0:
            augmented = drop_nodes(deepcopy(data))
        elif aug_type == 1:
            augmented = permute_edges(deepcopy(data))
        else:
            augmented = subgraph(deepcopy(data))

        augmented.batch = torch.zeros(augmented.x.shape[0]).long()
        return augmented
