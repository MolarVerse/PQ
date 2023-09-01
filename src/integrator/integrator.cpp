#include "integrator.hpp"

#include "constants.hpp"       // for _FS_TO_S_, _V_VERLET_VELOCITY_FACTOR_
#include "molecule.hpp"        // for Molecule
#include "simulationBox.hpp"   // for SimulationBox
#include "vector3d.hpp"        // for operator*, Vector3D

#include <vector>   // for vector

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
    simBox.applyPBC(positions);

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

    auto firstStepOfMolecule = [this, &simBox, &box](auto &molecule)
    {
        for (size_t i = 0, numberOfAtoms = molecule.getNumberOfAtoms(); i < numberOfAtoms; ++i)
        {
            integrateVelocities(molecule, i);
            integratePositions(molecule, i, simBox);
        }

        molecule.calculateCenterOfMass(box);
        molecule.setAtomForcesToZero();
    };

    std::ranges::for_each(simBox.getMolecules(), firstStepOfMolecule);
}

/**
 * @brief applies second half step of velocity verlet algorithm
 *
 * @param simBox
 */
void VelocityVerlet::secondStep(SimulationBox &simBox)
{
    auto secondStepOfMolecule = [this](auto &molecule)
    {
        for (size_t i = 0, numberOfAtoms = molecule.getNumberOfAtoms(); i < numberOfAtoms; ++i)
            integrateVelocities(molecule, i);
    };

    std::ranges::for_each(simBox.getMolecules(), secondStepOfMolecule);
}