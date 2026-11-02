"""Unit tests for AIMD mass/force/energy consistency (no Psi4 binary required)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Hartree ↔ eV (CODATA 2018); used to cross-check ASE/Psi4 conventions in docs/tests.
HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903


@pytest.fixture
def mond_aimd():
    """Import vendored mond.simulation.aimd from repo root."""
    # Ensure repo root wins over site-packages mond when both exist.
    repo_root = str(__import__("pathlib").Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    # Drop cached mond from site-packages if already imported without fix.
    for key in list(sys.modules):
        if key == "mond" or key.startswith("mond."):
            del sys.modules[key]
    from mond.simulation import aimd as aimd_mod
    from mond.simulation.constants import AMU_TO_EV_FS2_PER_A2

    return aimd_mod, AMU_TO_EV_FS2_PER_A2


def test_amu_to_ev_fs2_per_a2_reference(mond_aimd):
    _, amu_to_ev = mond_aimd
    # Derived from SI definitions (matches issue acceptance value).
    assert amu_to_ev == pytest.approx(103.642691909, rel=0, abs=1e-6)


def test_acceleration_units_one_amu(mond_aimd):
    """F/m with correct mass unit gives a = 1 Å/fs² for F = AMU_TO eV/Å on 1 amu."""
    aimd_mod, amu_to_ev = mond_aimd
    masses_amu = np.array([1.0])
    masses_ev = aimd_mod.masses_amu_to_ev_fs2_per_a2(masses_amu)
    force = np.array([[amu_to_ev, 0.0, 0.0]])
    acc = force / masses_ev[:, np.newaxis]
    assert acc[0, 0] == pytest.approx(1.0, rel=1e-12)


def test_hartree_to_ev_energy_conversion():
    """Methanol-scale total energy: Psi4 Hartree × factor ≈ ASE eV."""
    # Representative all-electron PBE total energy magnitude (Hartree), not a live Psi4 run.
    e_hartree = -115.48
    e_ev_expected = e_hartree * HARTREE_TO_EV
    # Bundled prerun step_0 is about -3142 eV for methanol — same ballpark.
    assert e_ev_expected == pytest.approx(-3142.6, rel=0.01)


def test_hartree_bohr_force_conversion():
    f_hartree_bohr = -0.01
    f_ev_ang = f_hartree_bohr * HARTREE_TO_EV / BOHR_TO_ANGSTROM
    assert f_ev_ang == pytest.approx(-0.514, rel=1e-3)


def test_velocity_verlet_step_constant_force_displacement(mond_aimd):
    """One Verlet half-step: Δx = 0.5 a dt² with mocked constant Psi4 forces."""
    aimd_mod, amu_to_ev = mond_aimd
    dt = 0.1  # fs
    masses_amu = np.array([1.0, 1.0])
    positions = np.zeros((2, 3))
    velocities = np.zeros((2, 3))
    forces = np.zeros((2, 3))
    forces[0, 0] = amu_to_ev  # → 1 Å/fs² for atom 0

    coll = MagicMock()
    coll.get_atom_symbols.return_value = ["H", "H"]

    def fake_psi4(**kwargs):
        pos = kwargs["positions"]
        return np.zeros_like(pos), 0.0

    with patch.object(aimd_mod, "get_energy_forces_psi4", side_effect=fake_psi4):
        new_pos, _, _, _ = aimd_mod.velocity_verlet_step(
            coll=coll,
            method_psi4="PBE",
            memory_psi4="1GB",
            basis_psi4="def2-SVP",
            scf_type_psi4="df",
            maxiter_psi4=50,
            positions=positions,
            velocities=velocities,
            forces=forces,
            masses=masses_amu,
            box_lengths=np.array([10.0, 10.0, 10.0]),
            dt=dt,
            periodic_boundary=False,
        )

    # x += 0.5 * 1 * dt^2
    expected_dx = 0.5 * dt**2
    assert new_pos[0, 0] == pytest.approx(expected_dx, rel=1e-12)
    assert new_pos[1, 0] == pytest.approx(0.0, abs=1e-15)


def test_velocity_verlet_no_wrap_when_non_periodic(mond_aimd):
    """Coordinates must not be folded into the box for gas-phase Psi4."""
    aimd_mod, amu_to_ev = mond_aimd
    positions = np.array([[9.5, 0.0, 0.0]])
    velocities = np.array([[2.0, 0.0, 0.0]])  # would cross box edge
    forces = np.zeros((1, 3))
    masses_amu = np.array([1.0])

    coll = MagicMock()
    coll.get_atom_symbols.return_value = ["H"]

    captured = {}

    def fake_psi4(**kwargs):
        captured["positions"] = kwargs["positions"].copy()
        return np.zeros((1, 3)), -1.0

    with patch.object(aimd_mod, "get_energy_forces_psi4", side_effect=fake_psi4):
        new_pos, _, _, _ = aimd_mod.velocity_verlet_step(
            coll=coll,
            method_psi4="PBE",
            memory_psi4="1GB",
            basis_psi4="def2-SVP",
            scf_type_psi4="df",
            maxiter_psi4=50,
            positions=positions,
            velocities=velocities,
            forces=forces,
            masses=masses_amu,
            box_lengths=np.array([10.0, 10.0, 10.0]),
            dt=0.5,
            periodic_boundary=False,
        )

    assert new_pos[0, 0] > 10.0
    assert captured["positions"][0, 0] > 10.0


def test_get_energy_forces_psi4_mocked_ase_units(mond_aimd):
    """Stored training labels use ASE eV / eV/Å from calculator outputs."""
    aimd_mod, _ = mond_aimd
    positions = np.array([[0.0, 0.0, 0.0]])
    coll = MagicMock()
    coll.get_atom_symbols.return_value = ["H"]

    e_hartree = -0.5
    e_ev = e_hartree * HARTREE_TO_EV
    forces_ev_ang = np.array([[0.1, 0.0, 0.0]])

    mock_atoms = MagicMock()
    mock_atoms.get_potential_energy.return_value = e_ev
    mock_atoms.get_forces.return_value = forces_ev_ang

    with patch.object(aimd_mod, "Atoms", return_value=mock_atoms):
        with patch.object(aimd_mod, "Psi4"):
            forces, e_pot = aimd_mod.get_energy_forces_psi4(
                coll=coll,
                positions=positions,
                method="PBE",
                memory="1GB",
                scf_type="df",
                basis="def2-SVP",
            )

    assert e_pot == pytest.approx(e_ev)
    assert np.allclose(forces, forces_ev_ang)
