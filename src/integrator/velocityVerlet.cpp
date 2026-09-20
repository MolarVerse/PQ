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

#include "globalTimer.hpp"
#include "simulationBox.hpp"

using namespace integrator;
using namespace molsys;

/**
 * @brief Construct a new Velocity Verlet::Velocity Verlet object
 *
 */
VelocityVerlet::VelocityVerlet() : Integrator("VelocityVerlet") {}

/**
 * @brief applies first half step of velocity verlet algorithm
 *
 * @param simulationBox
 */
void VelocityVerlet::firstStep(SimulationBox &simulationBox)
{
    auto _ = scopedTimer(TimerId::Integrator, "Velocity Verlet - First Step");

    auto integrate = [&simulationBox](auto &atom)
    {
        integrateVelocities(atom.get());
        integratePositions(atom.get(), simulationBox);
    };

    std::ranges::for_each(simulationBox.getAtoms(), integrate);

    const auto box = simulationBox.getBoxPtr();

    auto calculateCOM = [&box](auto &molecule)
    {
        molecule.calculateCenterOfMass(*box);
        molecule.setAtomForcesToZero();
    };

    std::ranges::for_each(simulationBox.getMolecules(), calculateCOM);
}

/**
 * @brief applies second half step of velocity verlet algorithm
 *
 * @param simulationBox
 */
void VelocityVerlet::secondStep(SimulationBox &simulationBox)
{
    auto _ = scopedTimer(TimerId::Integrator, "Velocity Verlet - Second Step");

    std::ranges::for_each(
        simulationBox.getAtoms(),
        [](const auto &atom) { integrateVelocities(atom.get()); }
    );
}
