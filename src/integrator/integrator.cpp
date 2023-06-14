#include "integrator.hpp"
#include "constants.hpp"

#include <vector>
#include <iostream>
#include <cmath>

using namespace std;

void Integrator::integrateVelocities(const double timestep, Molecule &molecule, const size_t i) const
{
    auto velocities = molecule.getAtomVelocity(i);
    const auto forces = molecule.getAtomForce(i);
    const auto mass = molecule.getMass(i);

    velocities += timestep * forces / mass * _V_VERLET_VELOCITY_FACTOR_;

    molecule.setAtomVelocities(i, velocities);
}

void Integrator::integratePositions(const double timestep, Molecule &molecule, const size_t i, const SimulationBox &simBox) const
{
    auto positions = molecule.getAtomPosition(i);
    const auto velocities = molecule.getAtomVelocity(i);

    positions += timestep * velocities * _FS_TO_S_;
    applyPBC(simBox, positions);

    molecule.setAtomPositions(i, positions);
}

void VelocityVerlet::firstStep(SimulationBox &simBox)
{
    const auto box = simBox._box.getBoxDimensions();

    for (auto &molecule : simBox._molecules)
    {
        const size_t numberOfAtoms = molecule.getNumberOfAtoms();

        for (size_t i = 0; i < numberOfAtoms; ++i)
        {
            integrateVelocities(_dt, molecule, i);
            integratePositions(_dt, molecule, i, simBox);
        }

        molecule.calculateCenterOfMass(box);
        molecule.setAtomForcesToZero();
    }
}

void VelocityVerlet::secondStep(SimulationBox &simBox)
{
    for (auto &molecule : simBox._molecules)
    {
        const size_t numberOfAtoms = molecule.getNumberOfAtoms();

        for (size_t i = 0; i < numberOfAtoms; ++i)
        {
            integrateVelocities(_dt, molecule, i);
        }
    }
}