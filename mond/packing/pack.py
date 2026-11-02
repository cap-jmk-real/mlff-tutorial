from copy import deepcopy
import numpy as np

from mond.docking import place_ions_of_salt_in_box_perturbed_docking_pose, get_docking_pose, get_perturbed_docking_pose
from mond.utils import random_pose, check_overlaps_vectorized
from mond.molecule import Molecule, MoleculeCollection



def pack_molecules_in_box_with_random_conformation_perturbed_to_existing_collection(
    coll,
    molecules_in_mixture, 
    probs, 
    bounding_box,
    max_molecules, 
    max_tries, 
    safety_distance, 
    perturbation,
    max_tries_anion_cation=1000,
    salt_safety_distance=0.1,
    salt_perturbation=0.1, 
    sampled_conformations=20, 
    rms_threshold=0.1, 
    num_threads=4, 
    minimize_energy=True, 
    continue_after_failed_mol=False
):
    """Allows to add a salt and random molecules with docking poses. Most complicated 
    function of all.

    Args:
        molecules_in_mixture (_type_): _description_
        probs (_type_): _description_
        bounding_box (_type_): _description_
        max_molecules (_type_): _description_
        max_tries (_type_): _description_
        safety_distance (_type_): _description_
        perturbation (_type_): _description_
        max_tries_anion_cation (int, optional): _description_. Defaults to 1000.
        salt_safety_distance (float, optional): _description_. Defaults to 0.1.
        salt_perturbation (float, optional): _description_. Defaults to 0.1.
        sampled_conformations (int, optional): _description_. Defaults to 20.
        rms_threshold (float, optional): _description_. Defaults to 0.1.
        num_threads (int, optional): _description_. Defaults to 4.
        minimize_energy (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    molec_1 = np.random.choice(molecules_in_mixture, p=probs)
    keep_trying = True
    counter = 1
    while counter <= max_tries and keep_trying:
        counter += 1
        
        if molec_1.is_salt:
            new_salt_coll = place_ions_of_salt_in_box_perturbed_docking_pose(
                        salt_coll = molec_1, 
                        bounding_box=bounding_box, 
                        perturbation=salt_perturbation, 
                        sampled_conformations=sampled_conformations, 
                        rms_threshold=rms_threshold, 
                        minimize_energy=minimize_energy, 
                        safety_distance=salt_safety_distance, 
                        max_tries_anion_cation=max_tries_anion_cation,
                        num_threads=num_threads
                        )
            keep_trying = False
        else: 
            new_coords = get_perturbed_docking_pose(
                                mol=molec_1.mol,
                                bounding_box=bounding_box,
                                perturbation=perturbation,
                                sampled_conformations=sampled_conformations,
                                rms_threshold=rms_threshold,
                                num_threads=num_threads,
                                minimize_energy=minimize_energy,
                            )
            if new_coords != None:
                keep_trying = False
    
    if molec_1.is_salt:
        coll = MoleculeCollection(coll.molecules + new_salt_coll.molecules)
    else: 
        molec_new = Molecule(
            deepcopy(molec_1.mol), new_coords, deepcopy(molec_1.radii)
        )  # assure no periodic overlap on negative values
        coll = MoleculeCollection(coll.molecules + [molec_new])
    for i in range(max_molecules - 1):

        molec_2 = np.random.choice(molecules_in_mixture, p=probs)
        keep_trying = True
        counter = 1
        while counter <= max_tries and keep_trying:
            counter += 1
            if molec_2.is_salt:
                new_salt_coll = place_ions_of_salt_in_box_perturbed_docking_pose(
                        salt_coll = molec_2, 
                        bounding_box=bounding_box, 
                        perturbation=salt_perturbation, 
                        sampled_conformations=sampled_conformations, 
                        rms_threshold=rms_threshold, 
                        minimize_energy=minimize_energy, 
                        safety_distance=salt_safety_distance, 
                        max_tries_anion_cation=max_tries_anion_cation,
                        num_threads=num_threads
                        )
                overlaps = check_overlaps_vectorized(
                    existing_centers=coll.get_atom_coords_list(),
                    existing_radii=coll.get_radii_list(),
                    new_centers=new_salt_coll.get_atom_coords_list(),
                    new_radii=new_salt_coll.get_radii_list(),
                    safety_distance=salt_safety_distance,
                )
                
            else:     
                new_coords = get_perturbed_docking_pose(
                            mol=molec_2.mol,
                            bounding_box=bounding_box,
                            perturbation=perturbation,
                            sampled_conformations=sampled_conformations,
                            rms_threshold=rms_threshold,
                            num_threads=num_threads,
                            minimize_energy=minimize_energy,
                        )
                if new_coords == None:
                    keep_trying = False
                    continue
                overlaps = check_overlaps_vectorized(
                    existing_centers=coll.get_atom_coords_list(),
                    existing_radii=coll.get_radii_list(),
                    new_centers=new_coords,
                    new_radii=molec_2.radii,
                    safety_distance=safety_distance,
                )
            # set new coordinates on the mol
            if overlaps == False:
                keep_trying = False
        if counter <= max_tries:
            if molec_2.is_salt: 
                coll = MoleculeCollection(coll.molecules+new_salt_coll.molecules)
            else: 
                molec_new = Molecule(
                    deepcopy(molec_2.mol), new_coords, deepcopy(molec_2.radii)
                )
                coll = MoleculeCollection(coll.molecules + [molec_new])
            print(f"{i+1} molecules added")
        
        else:
            if continue_after_failed_mol == True:
                continue
            else: 
                break
    return coll



def pack_molecules_in_box_with_random_conformation_perturbed(
    molecules_in_mixture, 
    probs, 
    bounding_box,
    max_molecules, 
    max_tries, 
    safety_distance, 
    perturbation,
    max_tries_anion_cation=1000,
    salt_safety_distance=0.1,
    salt_perturbation=0.1, 
    sampled_conformations=20, 
    rms_threshold=0.1, 
    num_threads=4, 
    minimize_energy=True, 
    continue_after_failed_mol=False
):
    """Allows to add a salt and random molecules with docking poses. Most complicated 
    function of all.

    Args:
        molecules_in_mixture (_type_): _description_
        probs (_type_): _description_
        bounding_box (_type_): _description_
        max_molecules (_type_): _description_
        max_tries (_type_): _description_
        safety_distance (_type_): _description_
        perturbation (_type_): _description_
        max_tries_anion_cation (int, optional): _description_. Defaults to 1000.
        salt_safety_distance (float, optional): _description_. Defaults to 0.1.
        salt_perturbation (float, optional): _description_. Defaults to 0.1.
        sampled_conformations (int, optional): _description_. Defaults to 20.
        rms_threshold (float, optional): _description_. Defaults to 0.1.
        num_threads (int, optional): _description_. Defaults to 4.
        minimize_energy (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    molec_1 = np.random.choice(molecules_in_mixture, p=probs)
    keep_trying = True
    counter = 1
    while counter <= max_tries and keep_trying:
        counter += 1
        
        if molec_1.is_salt:
            new_salt_coll = place_ions_of_salt_in_box_perturbed_docking_pose(
                        salt_coll = molec_1, 
                        bounding_box=bounding_box, 
                        perturbation=salt_perturbation, 
                        sampled_conformations=sampled_conformations, 
                        rms_threshold=rms_threshold, 
                        minimize_energy=minimize_energy, 
                        safety_distance=salt_safety_distance, 
                        max_tries_anion_cation=max_tries_anion_cation,
                        num_threads=num_threads
                        )
            keep_trying = False
        else: 
            new_coords = get_perturbed_docking_pose(
                                mol=molec_1.mol,
                                bounding_box=bounding_box,
                                perturbation=perturbation,
                                sampled_conformations=sampled_conformations,
                                rms_threshold=rms_threshold,
                                num_threads=num_threads,
                                minimize_energy=minimize_energy,
                            )
            if new_coords != None:
                keep_trying = False
    
    if molec_1.is_salt:
        coll = MoleculeCollection(new_salt_coll.molecules)
    else: 
        molec_new = Molecule(
            deepcopy(molec_1.mol), new_coords, deepcopy(molec_1.radii)
        )  # assure no periodic overlap on negative values
        coll = MoleculeCollection([molec_new])
    for i in range(max_molecules - 1):

        molec_2 = np.random.choice(molecules_in_mixture, p=probs)
        keep_trying = True
        counter = 1
        while counter <= max_tries and keep_trying:
            counter += 1
            if molec_2.is_salt:
                new_salt_coll = place_ions_of_salt_in_box_perturbed_docking_pose(
                        salt_coll = molec_2, 
                        bounding_box=bounding_box, 
                        perturbation=salt_perturbation, 
                        sampled_conformations=sampled_conformations, 
                        rms_threshold=rms_threshold, 
                        minimize_energy=minimize_energy, 
                        safety_distance=salt_safety_distance, 
                        max_tries_anion_cation=max_tries_anion_cation,
                        num_threads=num_threads
                        )
                overlaps = check_overlaps_vectorized(
                    existing_centers=coll.get_atom_coords_list(),
                    existing_radii=coll.get_radii_list(),
                    new_centers=new_salt_coll.get_atom_coords_list(),
                    new_radii=new_salt_coll.get_radii_list(),
                    safety_distance=salt_safety_distance,
                )
                
            else:     
                new_coords = get_perturbed_docking_pose(
                            mol=molec_2.mol,
                            bounding_box=bounding_box,
                            perturbation=perturbation,
                            sampled_conformations=sampled_conformations,
                            rms_threshold=rms_threshold,
                            num_threads=num_threads,
                            minimize_energy=minimize_energy,
                        )
                if new_coords == None:
                    keep_trying = False
                    continue
                overlaps = check_overlaps_vectorized(
                    existing_centers=coll.get_atom_coords_list(),
                    existing_radii=coll.get_radii_list(),
                    new_centers=new_coords,
                    new_radii=molec_2.radii,
                    safety_distance=safety_distance,
                )
            # set new coordinates on the mol
            if overlaps == False:
                keep_trying = False
        if counter <= max_tries:
            if molec_2.is_salt: 
                coll = MoleculeCollection(coll.molecules+new_salt_coll.molecules)
            else: 
                molec_new = Molecule(
                    deepcopy(molec_2.mol), new_coords, deepcopy(molec_2.radii)
                )
                coll = MoleculeCollection(coll.molecules + [molec_new])
            print(f"{i+1} molecules added")
        
        else:
            if continue_after_failed_mol == True:
                continue
            else: 
                break
    return coll


def pack_molecules_in_box_with_random_conformation(
    molecules_in_mixture, 
    probs, 
    bounding_box,
    max_molecules, 
    max_tries, 
    safety_distance, 
    sampled_conformations=20, 
    rms_threshold=0.1, 
    num_threads=4, 
    minimize_energy=True
):

    molec_1 = np.random.choice(molecules_in_mixture, p=probs)
    keep_trying = True
    counter = 1
    while counter <= max_tries and keep_trying:
        counter += 1
        new_coords =  get_docking_pose(
                            mol=molec_1.mol,
                            bounding_box=bounding_box,
                            sampled_conformations=sampled_conformations,
                            rms_threshold=rms_threshold,
                            num_threads=num_threads,
                            minimize_energy=minimize_energy,
                        )
        if new_coords != None:
            keep_trying = False
    molec_new = Molecule(
        deepcopy(molec_1.mol), new_coords, deepcopy(molec_1.radii)
    )  # assure no periodic overlap on negative values
    coll = MoleculeCollection([molec_new])
    for i in range(max_molecules - 1):
        print(f"{i+1} molecules added")
        molec_2 = np.random.choice(molecules_in_mixture, p=probs)
        keep_trying = True
        counter = 1
        while counter <= max_tries and keep_trying:
            counter += 1
            new_coords = get_docking_pose(
                            molec_2.mol,
                            bounding_box,
                            sampled_conformations=sampled_conformations,
                            rms_threshold=rms_threshold,
                            num_threads=num_threads,
                            minimize_energy=minimize_energy,
                        )
            if new_coords == None:
                keep_trying = False
                continue
            overlaps = check_overlaps_vectorized(
                existing_centers=coll.get_atom_coords_list(),
                existing_radii=coll.get_radii_list(),
                new_centers=new_coords,
                new_radii=molec_2.radii,
                safety_distance=safety_distance,
            )
            # set new coordinates on the mol
            if overlaps == False:
                keep_trying = False
        if counter <= max_tries:
            molec_new = Molecule(
                deepcopy(molec_2.mol), new_coords, deepcopy(molec_2.radii)
            )
            coll = MoleculeCollection(coll.molecules + [molec_new])
        else:
            break
    return coll


def pack_molecules_to_existing_collection(
    coll,
    molecules_in_mixture,
    probs,
    bounding_box,
    max_molecules,
    max_tries,
    safety_distance,
):

    molec_1 = np.random.choice(molecules_in_mixture, p=probs)
    keep_trying = True
    counter = 1
    while counter <= max_tries and keep_trying:
        counter += 1
        new_coords = random_pose(molec_1.coordinates, bounding_box=bounding_box)
        if new_coords != None:
            keep_trying = False
    molec_new = Molecule(
        deepcopy(molec_1.mol), new_coords, deepcopy(molec_1.radii)
    )  # assure no periodic overlap on negative values
    coll = MoleculeCollection(coll.molecules + [molec_new])
    molecules_added = 0
    for i in range(max_molecules - 1):
        print(f"{molecules_added} molecules added")
        molec_2 = np.random.choice(molecules_in_mixture, p=probs)
        keep_trying = True
        counter = 1
        while counter <= max_tries and keep_trying:
            counter += 1
            new_coords = random_pose(molec_2.coordinates, bounding_box=bounding_box)
            if new_coords == None:
                keep_trying = False
            overlaps = check_overlaps_vectorized(
                existing_centers=coll.get_atom_coords_list(),
                existing_radii=coll.get_radii_list(),
                new_centers=new_coords,
                new_radii=molec_2.radii,
                safety_distance=safety_distance,
            )
            # set new coordinates on the mol
            if overlaps == False:
                keep_trying = False
        if counter <= max_tries:
            molec_new = Molecule(
                deepcopy(molec_2.mol), new_coords, deepcopy(molec_2.radii)
            )
            coll = MoleculeCollection(coll.molecules + [molec_new])
            molecules_added += 1
        else:
            break
    return coll, molecules_added


def pack_molecules_in_box(
    molecules_in_mixture, probs, bounding_box, max_molecules, max_tries, safety_distance
):

    molec_1 = np.random.choice(molecules_in_mixture, p=probs)
    keep_trying = True
    counter = 1
    while counter <= max_tries and keep_trying:
        counter += 1
        new_coords = random_pose(molec_1.coordinates, bounding_box=bounding_box)
        if new_coords != None:
            keep_trying = False
    molec_new = Molecule(
        deepcopy(molec_1.mol), new_coords, deepcopy(molec_1.radii)
    )  # assure no periodic overlap on negative values
    coll = MoleculeCollection([molec_new])
    for i in range(max_molecules - 1):
        print(f"{i+1} molecules added")
        molec_2 = np.random.choice(molecules_in_mixture, p=probs)
        keep_trying = True
        counter = 1
        while counter <= max_tries and keep_trying:
            counter += 1
            new_coords = random_pose(molec_2.coordinates, bounding_box=bounding_box)
            if new_coords == None:
                keep_trying = False
                continue
            overlaps = check_overlaps_vectorized(
                existing_centers=coll.get_atom_coords_list(),
                existing_radii=coll.get_radii_list(),
                new_centers=new_coords,
                new_radii=molec_2.radii,
                safety_distance=safety_distance,
            )
            # set new coordinates on the mol
            if overlaps == False:
                keep_trying = False
        if counter <= max_tries:
            molec_new = Molecule(
                deepcopy(molec_2.mol), new_coords, deepcopy(molec_2.radii)
            )
            coll = MoleculeCollection(coll.molecules + [molec_new])
        else:
            break
    return coll
