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

#include "integrator.hpp"

#include "atom.hpp"
#include "simulationBox.hpp"

namespace integrator
{
    /**
     * @brief integrates the velocities of a single atom
     *
     * @param atom the atom whose velocities are to be integrated
     * @param timeStep the integration time step used for velocity updates
     */
    void Integrator::integrateVelocities(
        molsys::Atom *atom,
        const Time   &timeStep
    )
    {
        auto       velocity = atom->getVelocity();
        const auto force    = atom->getForce();
        const auto mass     = atom->getMass();

        velocity += timeStep * force / mass;

        atom->setVelocity(velocity);
    }

    /**
     * @brief integrates the positions of a single atom
     *
     * @param atom
     * @param simulationBox the simulation box to apply periodic boundary
     * conditions
     * @param timeStep the integration time step used for velocity updates
     */
    void Integrator::integratePositions(
        molsys::Atom                *atom,
        const molsys::SimulationBox &simulationBox,
        const Time                  &timeStep
    )
    {
        auto       position = atom->getPosition();
        const auto velocity = atom->getVelocity();

        position += timeStep * velocity;

        simulationBox.applyPBC(position);

        atom->setPosition(position);
    }

}   // namespace integrator
