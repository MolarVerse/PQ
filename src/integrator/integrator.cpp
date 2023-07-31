#include "integrator.hpp"

#include "constants.hpp"

#include <cmath>
#include <iostream>
#include <vector>

using namespace std;
using namespace simulationBox;
using namespace integrator;

/**
 * @brief integrates the velocities of a single atom
 *
 * @param molecule
 * @param i
 */
void Integrator::integrateVelocities(Molecule &molecule, const size_t i) const
{
    auto       velocities = molecule.getAtomVelocity(i);
    const auto forces     = molecule.getAtomForce(i);
    const auto mass       = molecule.getAtomMass(i);

    velocities += _dt * forces / mass * constants::_V_VERLET_VELOCITY_FACTOR_;

    molecule.setAtomVelocity(i, velocities);
}

/**
 * @brief integrates the positions of a single atom
 *
 * @param molecule
 * @param index
 * @param simBox
 */
void Integrator::integratePositions(Molecule &molecule, const size_t index, const SimulationBox &simBox) const
{
    auto       positions  = molecule.getAtomPosition(index);
    const auto velocities = molecule.getAtomVelocity(index);

    positions += _dt * velocities * constants::_FS_TO_S_;
    applyPBC(simBox, positions);

    molecule.setAtomPosition(index, positions);
}

/**
 * @brief applies first half step of velocity verlet algorithm
 *
 * @param simBox
 */
void VelocityVerlet::firstStep(SimulationBox &simBox)
{
    const auto box = simBox.getBoxDimensions();

    for (auto &molecule : simBox.getMolecules())
    {
        const size_t numberOfAtoms = molecule.getNumberOfAtoms();

        for (size_t i = 0; i < numberOfAtoms; ++i)
        {
            integrateVelocities(molecule, i);
            integratePositions(molecule, i, simBox);
        }

        molecule.calculateCenterOfMass(box);
        molecule.setAtomForcesToZero();
    }
}

/**
 * @brief applies second half step of velocity verlet algorithm
 *
 * @param simBox
 */
void VelocityVerlet::secondStep(SimulationBox &simBox)
{
    for (auto &molecule : simBox.getMolecules())
    {
        const size_t numberOfAtoms = molecule.getNumberOfAtoms();

        for (size_t i = 0; i < numberOfAtoms; ++i)
        {
            integrateVelocities(molecule, i);
        }
    }
}