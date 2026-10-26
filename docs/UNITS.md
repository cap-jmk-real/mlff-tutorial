# AIMD unit conventions (Psi4 → NPZ → ML training)

## Psi4 via ASE

The `mond.simulation.aimd.get_energy_forces_psi4` helper uses ASE’s `Psi4` calculator. ASE reports:

| Quantity | Unit |
|----------|------|
| Potential energy | eV |
| Forces | eV/Å |
| Positions | Å |

Psi4’s native output is in Hartree (energy) and Hartree/Bohr (gradient). ASE converts to eV and eV/Å before `get_potential_energy()` / `get_forces()` return. To compare with Psi4 printouts manually:

- \(E_\mathrm{eV} = E_\mathrm{Hartree} \times 27.211386245988\)
- \(F_{\mathrm{eV/Å}} = F_{\mathrm{Hartree/Bohr}} \times 27.211386245988 / 0.529177210903\)

## Molecular dynamics integration

Velocities are sampled and integrated in **Å/fs**. With forces in **eV/Å**, acceleration is **Å/fs²**, so masses must be in **eV·fs²/Å²**, not amu:

\[
\frac{\mathrm{eV}}{\mathrm{Å}} \div \frac{\mathrm{eV·fs²}}{\mathrm{Å²}} = \frac{\mathrm{Å}}{\mathrm{fs²}}
\]

Conversion: `1 amu = 103.642691909 eV·fs²/Å²` (`mond.simulation.constants.AMU_TO_EV_FS2_PER_A2`).

`initialize_velocities` already used this mass unit; `velocity_verlet_step` must use the same conversion when computing `forces / mass`.

## Stored dataset (`*.npz`)

Tutorial notebooks save:

- `positions` — Å  
- `forces` — eV/Å  
- `e_pot` — eV  
- `atom_types` — atomic numbers  

Tutorial 3 loads these tensors directly for MLFF training; no extra unit conversion is applied in the dataloader.

## Regenerating `methanol_aimd_data_prerun`

The bundled prerun set was produced with a `velocity_verlet_step` bug (dividing forces by amu instead of eV·fs²/Å²), so the **time evolution** of geometries is not a valid NVE trajectory at the intended timestep.

Per-frame `e_pot` and `forces` are still consistent Psi4 labels at the stored coordinates (energies are not scaled by ~103.6). **Do not** rescale energies or forces by 103.6; **regenerate** the trajectory with mond ≥ 0.2.6 (or this repo’s `pip install -e .`) if you need physically consistent AIMD sampling.
