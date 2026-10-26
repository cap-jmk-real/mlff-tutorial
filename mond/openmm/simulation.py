from sys import stdout

import openmm.app as app
import openmm as mm
import openmm.unit as openmmunit
from openmm.app import PDBFile

from openmmforcefields.generators import SMIRNOFFTemplateGenerator

from openff.toolkit import Molecule
from openff.toolkit import Topology as offTopology
from openff.units.openmm import to_openmm as offquantity_to_openmm
from openff.toolkit.topology import Topology
from openff.units import unit


def simulate_with_small_cutoff_LJPME(
    sdf_file: str,
    out_file_state: str,
    out_traj_name: str,
    box_vectors: list,
    simulation_steps: int,
    report_every_steps: int,
    cutoff: float,
    print_report_every_steps: int = 1000,
    temperature: float = 300,
    velocity_init_temperature: float = 300,
    timestep: float = 0.002,
    minimization_steps: int = 1000,
    force_field: str = "amber/protein.ff14SB.xml",
    water_model="amber/tip3p_standard.xml",
) -> None:
    """_summary_

    Args:
        pdb_file (str): Sample to start from
        out_file_state (str): save state to
        box_vectors (list): box vectors, i.e. [20, 20, 20] in angstrom.
        simulation_steps (int): number of simulation steps
        report_every_steps (int): Report every n steps
        cutoff (float): cutoff distance in nm
        print_report_every_steps (int, optional): Print report every n steps. Defaults to 1000.
        temperature (float, optional): In Kelvin. Defaults to 300.
        velocity_init_temperature (float, optional): In Kelvin for initialisation of the velocities. Defaults to 300.
        picoseconds (float, optional): Size of the timestep. Defaults to 0.002.
        minimization_steps (int, optional): Steps taken to minimize the sample prior to the simulation. Defaults to 1000.
        force_field (str, optional): Force field for protein. Not sure if this is necessary. Defaults to "amber/protein.ff14SB.xml".
        water_model (str, optional): Water model. Not sure if this is necessary. Defaults to "amber/tip3p_standard.xml".
    """

    assert (
        simulation_steps % report_every_steps == 0
    ), "simulation_steps and report_every_steps not divisible"
    mixture_path = sdf_file
    mixture = Molecule.from_file(mixture_path)

    smirnoff = SMIRNOFFTemplateGenerator(molecules=mixture)
    print(smirnoff.smirnoff_filename)
    ff = app.ForceField(force_field, water_model)
    ff.registerTemplateGenerator(smirnoff.generator)

    ligand_off_topology = offTopology.from_molecules(molecules=mixture)

    box_vectors = box_vectors * unit.nanometer
    ligand_off_topology.box_vectors = box_vectors
    ligand_omm_topology = ligand_off_topology.to_openmm()
    ligand_positions = []
    for molecule in mixture:
        ligand_positions += offquantity_to_openmm(molecule.conformers[0])
    modeller = app.Modeller(ligand_omm_topology, ligand_positions)

    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=app.LJPME,
        nonbondedCutoff=cutoff,
        constraints=app.HBonds,
    )
    integrator = mm.LangevinMiddleIntegrator(
        temperature * openmmunit.kelvin,
        1 / openmmunit.picosecond,
        timestep * openmmunit.picoseconds,
    )
    simulation = app.Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)
    print("Minimizing energy...")
    simulation.minimizeEnergy(maxIterations=minimization_steps)
    simulation.context.setVelocitiesToTemperature(
        velocity_init_temperature * openmmunit.kelvin
    )
    simulation.reporters.append(
        app.PDBReporter(f"{out_traj_name}.pdb", report_every_steps)
    )
    simulation.reporters.append(
        app.DCDReporter(f"{out_traj_name}.dcd", report_every_steps)
    )
    simulation.reporters.append(
        app.StateDataReporter(
            stdout,
            print_report_every_steps,
            step=True,
            potentialEnergy=True,
            temperature=True,
            speed=True,
        )
    )

    print("Running simulation...")
    simulation.step(simulation_steps)
    save_state(simulation=simulation, file_name=out_file_state)


def simulate_sample(
    sdf_file: str,
    out_file_state: str,
    out_traj_name: str,
    box_vectors: list,
    simulation_steps: int,
    report_every_steps: int,
    print_report_every_steps: int = 1000,
    temperature: float = 300,
    velocity_init_temperature: float = 300,
    timestep: float = 0.002,
    minimization_steps: int = 1000,
    force_field: str = "amber/protein.ff14SB.xml",
    water_model="amber/tip3p_standard.xml",
) -> None:
    """_summary_

    Args:
        pdb_file (str): Sample to start from
        out_file_state (str): save state to
        box_vectors (list): box vectors, i.e. [20, 20, 20] in angstrom.
        simulation_steps (int): number of simulation steps
        report_every_steps (int): Report every n steps
        print_report_every_steps (int, optional): Print report every n steps. Defaults to 1000.
        temperature (float, optional): In Kelvin. Defaults to 300.
        velocity_init_temperature (float, optional): In Kelvin for initialisation of the velocities. Defaults to 300.
        picoseconds (float, optional): Size of the timestep. Defaults to 0.002.
        minimization_steps (int, optional): Steps taken to minimize the sample prior to the simulation. Defaults to 1000.
        force_field (str, optional): Force field for protein. Not sure if this is necessary. Defaults to "amber/protein.ff14SB.xml".
        water_model (str, optional): Water model. Not sure if this is necessary. Defaults to "amber/tip3p_standard.xml".
    """

    assert (
        simulation_steps % report_every_steps == 0
    ), "simulation_steps and report_every_steps not divisible"
    mixture_path = sdf_file
    mixture = Molecule.from_file(mixture_path)

    smirnoff = SMIRNOFFTemplateGenerator(molecules=mixture)
    print(smirnoff.smirnoff_filename)
    ff = app.ForceField(force_field, water_model)
    ff.registerTemplateGenerator(smirnoff.generator)

    ligand_off_topology = offTopology.from_molecules(molecules=mixture)

    box_vectors = box_vectors * unit.nanometer
    ligand_off_topology.box_vectors = box_vectors
    ligand_omm_topology = ligand_off_topology.to_openmm()
    ligand_positions = []
    for molecule in mixture:
        ligand_positions += offquantity_to_openmm(molecule.conformers[0])
    modeller = app.Modeller(ligand_omm_topology, ligand_positions)

    system = ff.createSystem(
        modeller.topology, nonbondedMethod=app.PME, constraints=app.HBonds
    )
    integrator = mm.LangevinMiddleIntegrator(
        temperature * openmmunit.kelvin,
        1 / openmmunit.picosecond,
        timestep * openmmunit.picoseconds,
    )
    simulation = app.Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)
    print("Minimizing energy...")
    simulation.minimizeEnergy(maxIterations=minimization_steps)
    simulation.context.setVelocitiesToTemperature(
        velocity_init_temperature * openmmunit.kelvin
    )
    simulation.reporters.append(
        app.PDBReporter(f"{out_traj_name}.pdb", report_every_steps)
    )
    simulation.reporters.append(
        app.DCDReporter(f"{out_traj_name}.dcd", report_every_steps)
    )
    simulation.reporters.append(
        app.StateDataReporter(
            stdout,
            print_report_every_steps,
            step=True,
            potentialEnergy=True,
            temperature=True,
            speed=True,
        )
    )

    print("Running simulation...")
    simulation.step(simulation_steps)
    save_state(simulation=simulation, file_name=out_file_state)


def save_state(simulation, file_name: str) -> None:

    state = simulation.context.getState(getPositions=True)
    positions = state.getPositions()  # nm
    with open(file_name, "w") as f:
        PDBFile.writeFile(simulation.topology, positions, f)  # Angstrom
