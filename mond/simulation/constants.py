"""Unit constants for AIMD integration (ASE/Psi4 convention).

Psi4 via ASE reports:
  - potential energy in eV
  - forces in eV/Å

Velocities are integrated in Å/fs. With force in eV/Å and displacement in Å,
acceleration must be in Å/fs², so mass must be in eV·fs²/Å² (not amu).

Reference: 1 amu = 103.642691909 eV·fs²/Å²
  (derived from 1 eV = 1.602176634e-19 J, 1 amu = 1.66053906660e-27 kg,
   1 Å = 1e-10 m, 1 fs = 1e-15 s).
"""

# Boltzmann constant in eV/K
KB_EV_PER_K = 8.617333262145e-5

# Atomic mass unit → eV·fs²/Å² (mass unit for F = m a with F in eV/Å, a in Å/fs²)
AMU_TO_EV_FS2_PER_A2 = 103.642691909

# ASE/Psi4 energy and force units stored in trajectory NPZ files
ENERGY_UNIT = "eV"
FORCE_UNIT = "eV/Angstrom"
LENGTH_UNIT = "Angstrom"
TIME_UNIT = "fs"
VELOCITY_UNIT = "Angstrom/fs"
