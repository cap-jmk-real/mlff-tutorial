from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolfiles import MolFromXYZBlock
from rdkit.Chem.rdmolops import SanitizeMol
from openmm.app import PDBFile
from openmm.unit import nanometer
import numpy as np
from scipy.spatial.transform import Rotation

from mond.molecule import Molecule, MoleculeCollection


def get_splitted_collection_from_joined_smiles_mol(mol, coords, radii):
    """The simulation produces a joined smiles in rdkit. That I have to split again, then
    the simulation can be run.
    """

    smiles = Chem.MolToSmiles(mol)
    molecules = smiles.split(".")
    coords_index = 0
    molecule_1 = create_molecule_from_smiles(molecules[0])
    atoms = molecule_1.get_atom_symbols()
    molecule_1.set_mol_conf_coordinates(
        coords[coords_index : coords_index + len(atoms)]
    )
    molecule_1.set_radii(radii[coords_index : coords_index + len(atoms)])
    splitted_coll = MoleculeCollection([molecule_1])
    coords_index += len(atoms)
    for i in range(1, len(molecules)):
        molecule_1 = create_molecule_from_smiles(molecules[i])
        atoms = molecule_1.get_atom_symbols()
        molecule_1.set_mol_conf_coordinates(
            coords[coords_index : coords_index + len(atoms)]
        )
        molecule_1.set_radii(radii[coords_index : coords_index + len(atoms)])
        splitted_coll = MoleculeCollection(splitted_coll.molecules + [molecule_1])
        coords_index += len(atoms)
    return splitted_coll

def create_molecule_from_xyz_file(xyz_file:str): 
    
    mol = Chem.MolFromXYZFile(xyz_file)
    coords = get_coordinates(mol)
    radii = get_vdw_radii(mol)
    molec = Molecule(mol, coords, radii)
    return molec


def create_molecule_from_smiles(smiles: str):
    """Create a molecule class from a smiles string.

    Args:
        smiles (str): _description_

    Returns:
        _type_: _description_
    """
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)

    coords = get_coordinates(mol)
    radii = get_vdw_radii(mol)
    molec = Molecule(mol, coords, radii)
    return molec


def load_openmm_positions_from_pdb(pdb_file: str) -> list:
    """load openMM positions as list from a given PDB file. Returning a list ensures smooth Python operations.

    Args:
        pdb_file (str): Path to PDB file

    Returns:
        list: list with coordinates
    """

    pdb = PDBFile(pdb_file)
    positions = pdb.getPositions(asNumpy=True).value_in_unit(nanometer)
    return positions.tolist()


def convert_xyz_to_pdb(
    xyz_in: str, pdb_out: str, preoptimize: bool = False, sanitize: bool = True
) -> None:
    """converts an xyz file to a pdb file readable by OpenMM

    Args:
        xyz_in (str): _description_
        pdb_out (str): _description_
        preoptimize (bool, optional): Preoptimze the structure using UFF. Can lead to faulty structures. Defaults to False.
        sanitize (bool, optional): Sanitize Mol to ensure all atoms are saturated for OpenMM. Defaults to False.

    Raises:
        ValueError: _description_
    """

    mol = load_rdmol_from_xyz(xyz_file=xyz_in)

    if sanitize:
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            print("Sanitization failed:", e)

    if preoptimize:
        try:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.UFFOptimizeMolecule(mol)
        except Exception as e:
            print(e)

    w = Chem.rdmolfiles.PDBWriter(pdb_out)
    w.write(mol)
    w.close()


def load_rdmol_from_xyz(xyz_file: str):

    with open(xyz_file) as f:
        xyz_block = f.read()

    mol = MolFromXYZBlock(xyz_block)

    if mol is None:
        raise ValueError("Failed to read XYZ as RDKit Mol.")
    return mol


def get_vdw_radii(mol) -> list:
    """vdw radii in nanometer

    Args:
        mol (_type_): _description_

    Returns:
        list: _description_
    """

    periodic_table = Chem.GetPeriodicTable()
    radii = [
        periodic_table.GetRvdw(atom.GetSymbol()) for atom in mol.GetAtoms()
    ]  # in angstrom
    return radii

def get_vdw_radii_salt(ion) -> list:
    """vdw radii in nanometer

    Args:
        mol (_type_): _description_

    Returns:
        list: _description_
    """

    periodic_table = Chem.GetPeriodicTable()
    radii = [
        periodic_table.GetRvdw(atom.GetSymbol()) for atom in mol.GetAtoms()
    ]  # in angstrom
    return radii


def get_atomic_masses(atoms:list)->list: 
    
    periodic_table = Chem.GetPeriodicTable()
    return [periodic_table.GetAtomicWeight(symbol) for symbol in atoms]

def get_coordinates(mol) -> list:
    """Get coordinates in nm

    Args:
        mol (_type_): RDKit Mol

    Returns:
        list: coordinates in nm
    """
    try:
        conf = mol.GetConformer()
    except Exception as exc:
        confs = mol.GetConformers()
        conf = confs[0]
    coords = np.array(
        [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
    )  # in angstrom for openmm
    return coords.tolist()


def random_pose(mol_coords, bounding_box):
    """Randomly rotate and translate molecule within a bounding box"""
    R = Rotation.random().as_matrix()
    rotated = mol_coords @ R.T
    min_bounds = rotated.min(axis=0)
    max_bounds = rotated.max(axis=0)
    space = bounding_box - (max_bounds - min_bounds)
    if np.any(space <= 0):
        return None
    translation = np.random.rand(3) * space - min_bounds
    posed = rotated + translation
    return posed.tolist()


def check_overlaps_vectorized(
    existing_centers: list,
    existing_radii: list,
    new_centers: list,
    new_radii: list,
    safety_distance: float,
) -> bool:
    existing_centers = np.array(existing_centers)  # shape: (N, 3)
    existing_radii = np.array(existing_radii)  # shape: (N,)
    new_centers = np.array(new_centers)  # shape: (M, 3)
    new_radii = np.array(new_radii)  # shape: (M,)    # Compute pairwise distances between new and existing centers (M x N)
    diff = (
        new_centers[:, np.newaxis, :] - existing_centers[np.newaxis, :, :]
    )  # shape: (M, N, 3)
    dists = np.linalg.norm(diff, axis=2)

    # Compute pairwise sum of radii (M x N)
    radius_sums = (
        new_radii[:, np.newaxis] + existing_radii[np.newaxis, :] + safety_distance
    )  # shape: (M, N)
    # Check overlaps
    overlaps = dists < radius_sums  # shape: (M, N)
    any_overlap = np.any(overlaps, axis=1)  # Boolean array of length M
    sum_overlaps = np.sum(any_overlap)
    return sum_overlaps > 0


def check_bounding_box(coords: list, bounding_box: list):
    coords = np.array(coords)
    if np.sum(coords[:, 0] > bounding_box[0]) > 0:
        return False
    elif np.sum(coords[:, 1] > bounding_box[1]) > 0:
        return False
    elif np.sum(coords[:, 2] > bounding_box[2]) > 0:
        return False
    else:
        return True
