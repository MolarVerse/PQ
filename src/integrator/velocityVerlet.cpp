/*****************************************************************************
<GPL_HEADER>

    PQ
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

#include "velocityVerlet.hpp"

#include "simulationBox.hpp"

using namespace integrator;
using namespace simulationBox;

VelocityVerlet::VelocityVerlet() : Integrator("VelocityVerlet"){};

/**
 * @brief applies first half step of velocity verlet algorithm
 *
 * @param simBox
 */
void VelocityVerlet::firstStep(SimulationBox &simBox)
{
    startTimingsSection("Velocity Verlet - First Step");

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

    stopTimingsSection("Velocity Verlet - First Step");
}

/**
 * @brief applies second half step of velocity verlet algorithm
 *
 * @param simBox
 */
void VelocityVerlet::secondStep(SimulationBox &simBox)
{
    startTimingsSection("Velocity Verlet - Second Step");

    std::ranges::for_each(
        simBox.getAtoms(),
        [this](auto atom) { integrateVelocities(atom.get()); }
    );

    stopTimingsSection("Velocity Verlet - Second Step");
}