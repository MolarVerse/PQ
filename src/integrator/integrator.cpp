/*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

#include "integrator.hpp"

#include "atom.hpp"                                  // for Atom
#include "constants/conversionFactors.hpp"           // for _FS_TO_S_
#include "constants/internalConversionFactors.hpp"   // for _V_VERLET_VELOCITY_FACTOR_
#include "simulationBox.hpp"                         // for SimulationBox
#include "timingsSettings.hpp"                       // for TimingsSettings
#include "vector3d.hpp"                              // for operator*, Vector3D

#include <algorithm>    // for __for_each_fn, for_each
#include <functional>   // for identity

using namespace integrator;

/**
 * @brief integrates the velocities of a single atom
 *
 * @param molecule
 * @param index
 */
void Integrator::integrateVelocities(simulationBox::Atom *atom) const
{
    auto       velocity = atom->getVelocity();
    const auto force    = atom->getForce();
    const auto mass     = atom->getMass();

    velocity += settings::TimingsSettings::getTimeStep() * force / mass * constants::_V_VERLET_VELOCITY_FACTOR_;

    atom->setVelocity(velocity);
}

/**
 * @brief integrates the positions of a single atom
 *
 * @param molecule
 * @param index
 * @param simBox
 */
void Integrator::integratePositions(simulationBox::Atom *atom, const simulationBox::SimulationBox &simBox) const
{
    auto       position = atom->getPosition();
    const auto velocity = atom->getVelocity();

    position += settings::TimingsSettings::getTimeStep() * velocity * constants::_FS_TO_S_;
    simBox.applyPBC(position);

    atom->setPosition(position);
}

/**
 * @brief applies first half step of velocity verlet algorithm
 *
 * @param simBox
 */
void VelocityVerlet::firstStep(simulationBox::SimulationBox &simBox)
{

    auto integrate = [this, &simBox](auto &atom)
    {
        integrateVelocities(atom.get());
        integratePositions(atom.get(), simBox);
    };

    std::ranges::for_each(simBox.getAtoms(), integrate);

    const auto box = simBox.getBoxPtr();

    auto calculateCOM = [&box](auto &molecule)
    {
        molecule.calculateCenterOfMass(*box);
        molecule.setAtomForcesToZero();
    };

    std::ranges::for_each(simBox.getMolecules(), calculateCOM);
}

/**
 * @brief applies second half step of velocity verlet algorithm
 *
 * @param simBox
 */
void VelocityVerlet::secondStep(simulationBox::SimulationBox &simBox)
{
    std::ranges::for_each(simBox.getAtoms(), [this](auto atom) { integrateVelocities(atom.get()); });
}