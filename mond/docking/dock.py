from copy import deepcopy
import random
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolAlign
from rdkit.Chem import rdForceFieldHelpers
from rdkit.Chem import SDWriter

from mond.utils import get_coordinates, random_pose, check_overlaps_vectorized
from mond.molecule import SaltCollection

def place_ions_of_salt_in_box_perturbed_docking_pose(
    salt_coll, 
    bounding_box, 
    perturbation, 
    sampled_conformations, 
    rms_threshold, 
    minimize_energy, 
    safety_distance, 
    max_tries_anion_cation, 
    num_threads
    ): 

    anion_molecs = salt_coll.anions
    cation_molecs = salt_coll.cations
    overlaps=True
    counter = 0

    while overlaps and counter <= max_tries_anion_cation: 
        counter +=1
        coords_anions = []
        radii_anions = []
        new_anion_list = []
        for anion in anion_molecs: 
            coords_anion = get_perturbed_docking_pose(
                mol = anion.mol,
                bounding_box = bounding_box,
                perturbation = perturbation,
                sampled_conformations=sampled_conformations,
                rms_threshold=rms_threshold,
                num_threads=num_threads,
                minimize_energy=minimize_energy,
            )
            radii_anion=anion.radii
            new_anion = deepcopy(anion)
            new_anion.set_mol_conf_coordinates(coords_anion)
            new_anion_list.append(new_anion)
            coords_anions += coords_anion
            radii_anions += radii_anion   
        coords_cations = []
        radii_cations = []
        new_cation_list = []
        for cation in cation_molecs:     
            coords_cation = get_perturbed_docking_pose(
                mol = cation.mol,
                bounding_box = bounding_box,
                perturbation = perturbation,
                sampled_conformations=sampled_conformations,
                rms_threshold=rms_threshold,
                num_threads=num_threads,
                minimize_energy=minimize_energy,
            )
            radii_cation = cation.radii
            new_cation = deepcopy(cation)
            new_cation.set_mol_conf_coordinates(coords_cation)
            new_cation_list.append(new_cation)
            coords_cations += coords_cation
            radii_cations += radii_cation

        overlaps = check_overlaps_vectorized(
            existing_centers=coords_cations, 
            existing_radii=radii_cations,
            new_centers=coords_anions, 
            new_radii=radii_anions,
            safety_distance=safety_distance
        ) 
        
    if counter <= max_tries_anion_cation: 
        return SaltCollection(anions=new_anion_list, cations=new_cation_list)
    else: 
        return None

def get_perturbed_docking_pose(
    mol,
    bounding_box,
    perturbation,
    sampled_conformations=20,
    rms_threshold=0.1,
    num_threads=4,
    minimize_energy=True,
):

    mol = get_conformation_mol(
        mol=mol,
        sampled_conformations=sampled_conformations,
        rms_threshold=rms_threshold,
        num_threads=num_threads,
        minimize_energy=minimize_energy,
    )
    mol_coords = get_coordinates(mol)
    mol_coords = random_pose_with_perturbation(
        mol_coords=mol_coords, bounding_box=bounding_box, perturbation=perturbation
    )
    return mol_coords


def random_pose_with_perturbation(perturbation, mol_coords, bounding_box):
    mol_coords = np.array(mol_coords)
    perturbation_mat = random_array = np.random.uniform(
        -perturbation, perturbation, size=np.array(mol_coords).shape
    )
    mol_coords = mol_coords + perturbation_mat
    mol_coords = random_pose(mol_coords=mol_coords, bounding_box=bounding_box)
    return mol_coords


def get_docking_pose(
    mol,
    bounding_box,
    sampled_conformations=20,
    rms_threshold=0.1,
    num_threads=4,
    minimize_energy=True,
):
    """Randomly rotate and translate a random conformation of a molecule. Sampled with RDKit"""

    mol = get_conformation_mol(
        mol=mol,
        sampled_conformations=sampled_conformations,
        rms_threshold=rms_threshold,
        num_threads=num_threads,
        minimize_energy=minimize_energy,
    )
    mol_coords = get_coordinates(mol)
    mol_coords = random_pose(mol_coords=mol_coords, bounding_box=bounding_box)
    return mol_coords


def get_conformation_mol(
    mol,
    sampled_conformations=20,
    rms_threshold=0.1,
    num_threads=4,
    minimize_energy=True,
):
    # Konformer-Sampling mit ETKDG (empirisch + knowledge-based)
    params = AllChem.ETKDGv3()
    params.numThreads = num_threads
    params.pruneRmsThresh = rms_threshold
    try: 
        conformer_ids = AllChem.EmbedMultipleConfs(
            mol, numConfs=sampled_conformations, params=params
        )

        if minimize_energy:
            for conf_id in conformer_ids:
                AllChem.UFFOptimizeMolecule(mol, confId=conf_id)

        sampled_id = random.choice(list(conformer_ids))
        new_mol = Chem.Mol(mol)
        conf = mol.GetConformer(sampled_id)
        new_mol.RemoveAllConformers()
        new_mol.AddConformer(conf, assignId=True)
    except Exception as exc: 
        print("Could not sample conformer for molecule")
        new_mol = Chem.Mol(mol) #return old conformer

    return new_mol
