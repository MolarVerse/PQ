#include "integrator.hpp"
#include "constants.hpp"

#include <vector>
#include <iostream>
#include <cmath>

using namespace std;

void VelocityVerlet::firstStep(SimulationBox &simulationBox, const Timings &timings)
{
    auto positions = Vec3D();
    auto velocities = Vec3D();
    auto forces = Vec3D();

    const auto box = simulationBox._box.getBoxDimensions();

    const auto timeStep = timings.getTimestep();

    for (auto &molecule : simulationBox._molecules)
    {
        const size_t numberOfAtoms = molecule.getNumberOfAtoms();

        for (size_t i = 0; i < numberOfAtoms; ++i)
        {
            positions = molecule.getAtomPositions(i);
            velocities = molecule.getAtomVelocities(i);
            forces = molecule.getAtomForces(i);
            const auto mass = molecule.getMass(i);

            velocities[0] += timeStep * forces[0] / mass * _V_VERLET_VELOCITY_FACTOR_;
            velocities[1] += timeStep * forces[1] / mass * _V_VERLET_VELOCITY_FACTOR_;
            velocities[2] += timeStep * forces[2] / mass * _V_VERLET_VELOCITY_FACTOR_;

            molecule.setAtomVelocities(i, velocities);

            positions[0] += timeStep * velocities[0] * _FS_TO_S_;
            positions[1] += timeStep * velocities[1] * _FS_TO_S_;
            positions[2] += timeStep * velocities[2] * _FS_TO_S_;

            positions[0] -= box[0] * round(positions[0] / box[0]);
            positions[1] -= box[1] * round(positions[1] / box[1]);
            positions[2] -= box[2] * round(positions[2] / box[2]);

            molecule.setAtomPositions(i, positions);
        }

        molecule.calculateCenterOfMass(box);
        molecule.setAtomForcesToZero();
    }
}

void VelocityVerlet::secondStep(SimulationBox &simulationBox, const Timings &timings)
{
    auto velocities = Vec3D();
    auto forces = Vec3D();

    const auto timeStep = timings.getTimestep();

    for (auto &molecule : simulationBox._molecules)
    {
        const size_t numberOfAtoms = molecule.getNumberOfAtoms();

        for (size_t i = 0; i < numberOfAtoms; ++i)
        {
            velocities = molecule.getAtomVelocities(i);
            forces = molecule.getAtomForces(i);
            const auto mass = molecule.getMass(i);

            velocities[0] += timeStep * forces[0] / mass * _V_VERLET_VELOCITY_FACTOR_;
            velocities[1] += timeStep * forces[1] / mass * _V_VERLET_VELOCITY_FACTOR_;
            velocities[2] += timeStep * forces[2] / mass * _V_VERLET_VELOCITY_FACTOR_;

            molecule.setAtomVelocities(i, velocities);
        }
    }
}