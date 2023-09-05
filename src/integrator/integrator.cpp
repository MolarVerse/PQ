#include "integrator.hpp"

#include "constants.hpp"         // for _FS_TO_S_, _V_VERLET_VELOCITY_FACTOR_
#include "molecule.hpp"          // for Molecule
#include "simulationBox.hpp"     // for SimulationBox
#include "timingsSettings.hpp"   // for TimingsSettings
#include "vector3d.hpp"          // for operator*, Vector3D

#include <algorithm>    // for __for_each_fn, for_each
#include <functional>   // for identity

using namespace integrator;

/**
 * @brief integrates the velocities of a single atom
 *
 * @param molecule
 * @param index
 */
void Integrator::integrateVelocities(simulationBox::Molecule &molecule, const size_t index) const
{
    auto       velocities = molecule.getAtomVelocity(index);
    const auto forces     = molecule.getAtomForce(index);
    const auto mass       = molecule.getAtomMass(index);

    velocities += settings::TimingsSettings::getTimeStep() * forces / mass * constants::_V_VERLET_VELOCITY_FACTOR_;

    molecule.setAtomVelocity(index, velocities);
}

/**
 * @brief integrates the positions of a single atom
 *
 * @param molecule
 * @param index
 * @param simBox
 */
void Integrator::integratePositions(simulationBox::Molecule            &molecule,
                                    const size_t                        index,
                                    const simulationBox::SimulationBox &simBox) const
{
    auto       positions  = molecule.getAtomPosition(index);
    const auto velocities = molecule.getAtomVelocity(index);

    positions += settings::TimingsSettings::getTimeStep() * velocities * constants::_FS_TO_S_;
    simBox.applyPBC(positions);

    molecule.setAtomPosition(index, positions);
}

/**
 * @brief applies first half step of velocity verlet algorithm
 *
 * @param simBox
 */
void VelocityVerlet::firstStep(simulationBox::SimulationBox &simBox)
{
    const auto box = simBox.getBoxDimensions();

    auto firstStepForMolecule = [this, &simBox, &box](auto &molecule)
    {
        for (size_t i = 0, numberOfAtoms = molecule.getNumberOfAtoms(); i < numberOfAtoms; ++i)
        {
            integrateVelocities(molecule, i);
            integratePositions(molecule, i, simBox);
        }

        molecule.calculateCenterOfMass(box);
        molecule.setAtomForcesToZero();
    };

    std::ranges::for_each(simBox.getMolecules(), firstStepForMolecule);
}

/**
 * @brief applies second half step of velocity verlet algorithm
 *
 * @param simBox
 */
void VelocityVerlet::secondStep(simulationBox::SimulationBox &simBox)
{
    auto secondStepForMolecule = [this](auto &molecule)
    {
        for (size_t i = 0, numberOfAtoms = molecule.getNumberOfAtoms(); i < numberOfAtoms; ++i)
            integrateVelocities(molecule, i);
    };

    std::ranges::for_each(simBox.getMolecules(), secondStepForMolecule);
}