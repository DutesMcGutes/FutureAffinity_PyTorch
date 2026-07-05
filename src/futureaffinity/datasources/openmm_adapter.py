from __future__ import annotations

from dataclasses import dataclass

from futureaffinity.datasources.base import PoseResult

_INSTALL_HINT = (
    "OpenMMSimulationSource requires the 'openmm' package (pip install openmm). This is a "
    "real integration point, not part of the default install -- it isn't needed to train or "
    "run the core model."
)


@dataclass
class OpenMMSimulationSource:
    """A short, real (vacuum, implicit-solvent-free) OpenMM Langevin trajectory.

    Takes a prepared PDB file (real bonds/topology required -- FutureAffinity's
    per-token coordinate representation alone isn't enough to build an OpenMM
    System) and returns a handful of snapshots along the trajectory with their
    potential energies, as a stand-in for the "MD refinement" data source in
    the synthetic-supervision pipeline.
    """

    temperature_kelvin: float = 300.0
    friction_per_ps: float = 1.0
    timestep_fs: float = 2.0

    def simulate_file(self, pdb_path: str, num_steps: int = 100, snapshot_every: int = 20) -> list[PoseResult]:
        try:
            import openmm
            import openmm.app as app
            import openmm.unit as unit
        except ImportError as error:
            raise RuntimeError(_INSTALL_HINT) from error

        pdb = app.PDBFile(pdb_path)
        forcefield = app.ForceField("amber14-all.xml")
        system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff)
        integrator = openmm.LangevinMiddleIntegrator(
            self.temperature_kelvin * unit.kelvin,
            self.friction_per_ps / unit.picosecond,
            self.timestep_fs * unit.femtosecond,
        )
        simulation = app.Simulation(pdb.topology, system, integrator)
        simulation.context.setPositions(pdb.positions)
        simulation.minimizeEnergy()

        results = []
        steps_taken = 0
        while steps_taken < num_steps:
            simulation.step(snapshot_every)
            steps_taken += snapshot_every
            state = simulation.context.getState(getPositions=True, getEnergy=True)
            coords = _positions_to_tensor(state.getPositions(asNumpy=True))
            energy = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
            results.append(PoseResult(coords=coords, energy=float(energy), source="openmm"))
        return results


def _positions_to_tensor(positions):
    import torch

    return torch.tensor(positions._value, dtype=torch.float32)
