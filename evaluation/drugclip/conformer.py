"""SMILES -> single deterministic 3D conformer (ETKDGv3 + MMFF), heavy atoms only."""
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")


def smiles_to_conformers(smi, n_conf=1, seed=42):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    ps = AllChem.ETKDGv3()
    ps.randomSeed = seed
    ps.maxIterations = 200
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conf, params=ps)
    if len(cids) == 0:
        ps.useRandomCoords = True
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conf, params=ps)
    if len(cids) == 0:
        return None
    try:
        AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=200)
    except Exception:
        pass
    mol = Chem.RemoveHs(mol)
    atoms = np.array([a.GetSymbol() for a in mol.GetAtoms()])
    coords = [np.asarray(mol.GetConformer(c).GetPositions(), dtype=np.float32)
              for c in range(mol.GetNumConformers())]
    return atoms, coords
