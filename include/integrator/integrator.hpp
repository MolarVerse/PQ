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

#ifndef INTEGRATOR__INTEGRATOR_HPP
#define INTEGRATOR__INTEGRATOR_HPP

#include "unit_types.hpp"

namespace molsys
{
    class Atom;            // forward declaration
    class SimulationBox;   // forward declaration
}   // namespace molsys

namespace integrator
{
    /**
     * @class Integrator
     *
     * @brief Integrator is a base class for all integrators
     *
     */
    class Integrator
    {
       public:
        Integrator()          = default;
        virtual ~Integrator() = default;

        virtual void firstStep(
            molsys::SimulationBox &simulationBox,
            const Time            &timeStep
        ) = 0;
        virtual void secondStep(
            molsys::SimulationBox &simulationBox,
            const Time            &timeStep
        ) = 0;

        static void integrateVelocities(molsys::Atom *, const Time &timeStep);
        static void integratePositions(
            molsys::Atom *,
            const molsys::SimulationBox &,
            const Time &timeStep
        );
    };

}   // namespace integrator

#endif   // INTEGRATOR__INTEGRATOR_HPP
