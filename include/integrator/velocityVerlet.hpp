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

#ifndef _VELOCITY_VERLET_HPP_

#define _VELOCITY_VERLET_HPP_

#include "integrator.hpp"
#include "typeAliases.hpp"

namespace integrator
{
    constexpr auto vvFirstStepMsg  = "Velocity Verlet - First Step";
    constexpr auto vvSecondStepMsg = "Velocity Verlet - Second Step";

    /**
     * @class VelocityVerlet inherits Integrator
     *
     * @brief VelocityVerlet is a class for velocity verlet integrator
     *
     */
    class VelocityVerlet : public Integrator
    {
       public:
        explicit VelocityVerlet();

        void firstStep(pq::SimBox &) override;
        void initFirstStep(pq::SimBox &);
        void finalizeFirstStep(pq::SimBox &);

        void secondStep(pq::SimBox &) override;
        void initSecondStep(pq::SimBox &);
        void finalizeSecondStep(pq::SimBox &);
    };

}   // namespace integrator

#endif   // _VELOCITY_VERLET_HPP_