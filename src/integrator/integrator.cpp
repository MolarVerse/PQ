#include "integrator.hpp"
#include "constants.hpp"

#include <vector>
#include <iostream>
#include <cmath>

using namespace std;

void VelocityVerlet::firstStep(SimulationBox &simulationBox, const Timings &timings)
{
    auto positions = vector<double>(3);
    auto velocities = vector<double>(3);
    auto forces = vector<double>(3);

    auto timeStep = timings.getTimestep();

    auto box = simulationBox._box.getBoxDimensions();

    for (auto &molecule : simulationBox._molecules)
    {
        for (int i = 0; i < molecule.getNumberOfAtoms(); i++)
        {
            molecule.getAtomPositions(i, positions);
            molecule.getAtomVelocities(i, velocities);
            molecule.getAtomForces(i, forces);
            auto mass = molecule.getMass(i);

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
        molecule.resetAtomForces();
    }
}

void VelocityVerlet::secondStep(SimulationBox &simulationBox, const Timings &timings)
{
    auto velocities = vector<double>(3);
    auto forces = vector<double>(3);

    auto timeStep = timings.getTimestep();

    for (auto &molecule : simulationBox._molecules)
    {
        for (int i = 0; i < molecule.getNumberOfAtoms(); i++)
        {
            molecule.getAtomVelocities(i, velocities);
            molecule.getAtomForces(i, forces);
            auto mass = molecule.getMass(i);

            velocities[0] += timeStep * forces[0] / mass * _V_VERLET_VELOCITY_FACTOR_;
            velocities[1] += timeStep * forces[1] / mass * _V_VERLET_VELOCITY_FACTOR_;
            velocities[2] += timeStep * forces[2] / mass * _V_VERLET_VELOCITY_FACTOR_;

            molecule.setAtomVelocities(i, velocities);
        }
    }
}