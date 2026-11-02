
import numpy as np

from ase import Atoms
from ase.calculators.psi4 import Psi4

from mond.simulation.constants import AMU_TO_EV_FS2_PER_A2, KB_EV_PER_K


def masses_amu_to_ev_fs2_per_a2(masses_amu: np.ndarray) -> np.ndarray:
    """Convert atomic masses from amu to eV·fs²/Å² for F/m with F in eV/Å."""
    return np.asarray(masses_amu, dtype=float) * AMU_TO_EV_FS2_PER_A2


def velocity_verlet_step(
    coll,
    method_psi4,
    memory_psi4,
    basis_psi4,
    scf_type_psi4,
    maxiter_psi4,
    positions,
    velocities,
    forces,
    masses,
    box_lengths,
    dt,
    periodic_boundary,
):
    """
    Velocity Verlet integrator for gas-phase (or PBC) AIMD with Psi4 forces.

    Units
    -----
    positions : Å
    velocities : Å/fs
    forces : eV/Å (ASE Psi4 calculator)
    masses : amu (same convention as ``get_atomic_masses``)
    dt : fs
    potential_energy return value : eV
    """
    masses_ev_fs2_per_a2 = masses_amu_to_ev_fs2_per_a2(masses)
    acc = forces / masses_ev_fs2_per_a2[:, np.newaxis]

    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    positions += velocities * dt + 0.5 * acc * dt**2

    # Psi4/ASE setups used here are non-periodic; wrapping coordinates would
    # break isolated-molecule geometries and is only applied when PBC is on.
    if periodic_boundary:
        positions = positions % box_lengths

    new_forces, potential_energy = get_energy_forces_psi4(
        coll=coll,
        positions=positions,
        method=method_psi4,
        memory=memory_psi4,
        basis=basis_psi4,
        scf_type=scf_type_psi4,
        maxiter=maxiter_psi4,
    )
    new_acc = new_forces / masses_ev_fs2_per_a2[:, np.newaxis]
    velocities = velocities + 0.5 * (acc + new_acc) * dt
    assert np.sum(np.isnan(velocities)) == 0, "Velocities contain NaN values"
    return positions, velocities, new_forces, potential_energy


def apply_pbc(displacement, box_length):
    return displacement - box_length * np.round(displacement / box_length)


def noose_hover_chain():
    raise NotImplementedError


def get_energy_forces_psi4(
    coll,
    positions,
    method,
    memory,
    scf_type,
    basis,
    charge=None,
    multiplicity=None,
    reference=None,
    maxiter=None,
):
    """
    Single-point energy and forces from Psi4 via ASE.

    Returns forces in eV/Å and potential energy in eV (ASE standard units).
    """
    atoms = Atoms(
        symbols=coll.get_atom_symbols(),
        positions=positions,
    )
    calc = Psi4(
        atoms=atoms,
        method=method,
        memory=memory,
        basis=basis,
        charge=charge,
        multiplicity=multiplicity,
        reference=reference,
        scf_type=scf_type,
    )

    calc.set_options({"reference": reference})
    calc.set(maxiter=maxiter)
    atoms.calc = calc

    pot_energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    return forces, pot_energy


def get_energy_forces_quantum_espresso():
    raise NotImplementedError


def pulling_force(
    positions: np.ndarray,
    box_lengths: np.ndarray,
    masses: np.ndarray,
    k_pull: float,
) -> np.ndarray:
    """
    Compute harmonic restoring forces that pull each particle toward the center of the box.

    Parameters:
        positions (np.ndarray): Array of shape (N, 3) with atomic positions in Å.
        box_length (np.ndarray): Array of shape (3,) with box lengths in Å.
        k_pull (float): Force constant (in eV/Å²).

    Returns:
        np.ndarray: Array of shape (N, 3) with forces in eV/Å.
    """
    positions = np.asarray(positions)
    box_lengths = np.asarray(box_lengths)
    masses = np.asarray(masses)

    center = box_lengths / 2

    total_mass = masses.sum()
    com = np.sum(masses[:, None] * positions, axis=0) / total_mass

    displacement = com - center

    total_force = -k_pull * displacement

    forces = (masses[:, None] / total_mass) * total_force

    return forces


def initialize_velocities(atom_masses, temp, remove_drift=True) -> list:
    N = len(atom_masses)
    atom_masses = np.array(atom_masses)
    masses_eV_fs2_per_A2 = masses_amu_to_ev_fs2_per_a2(atom_masses)

    sigma = np.sqrt(KB_EV_PER_K * temp / masses_eV_fs2_per_A2)

    velocities = np.random.normal(0.0, sigma[:, np.newaxis], size=(N, 3))

    if remove_drift:
        total_mass = masses_eV_fs2_per_A2.sum()
        com_velocity = (
            np.sum(velocities * masses_eV_fs2_per_A2[:, np.newaxis], axis=0) / total_mass
        )
        velocities -= com_velocity

    return velocities


def init_traj(traj_file: str) -> None:
    with open(traj_file, "w") as f:
        pass


def append_to_traj(traj_file: str, positions: list, symbols: list) -> None:
    N = len(symbols)
    with open(traj_file, "a") as f:
        f.write(f"{N}\n")
        f.write(f"AIMD trajectory point of mond\n")
        for sym, pos in zip(symbols, positions):
            f.write(f"{sym} {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}\n")
