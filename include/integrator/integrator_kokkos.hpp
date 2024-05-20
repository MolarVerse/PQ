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

#ifndef _KOKKOS_INTEGRATOR_HPP_

#define _KOKKOS_INTEGRATOR_HPP_

#include <Kokkos_DualView.hpp>

#include "simulationBox.hpp"
#include "simulationBox_kokkos.hpp"

namespace integrator
{
    /**
     * @class KokkosVelocityVerlet
     *
     * @brief KokkosVelocityVerlet is a class for the velocity Verlet integrator
     *
     */
    class KokkosVelocityVerlet
    {
       private:
        Kokkos::DualView<double> _dt;
        Kokkos::DualView<double> _velocityFactor;
        Kokkos::DualView<double> _timeFactor;

       public:
        explicit KokkosVelocityVerlet(
            const double dt,
            const double velocityFactor,
            const double timeFactor
        );

        KokkosVelocityVerlet()  = default;
        ~KokkosVelocityVerlet() = default;

        void firstStep(
            simulationBox::SimulationBox       &simBox,
            simulationBox::KokkosSimulationBox &kokkosSimBox
        );

        void secondStep(
            simulationBox::SimulationBox       &simBox,
            simulationBox::KokkosSimulationBox &kokkosSimBox
        );

        KOKKOS_FUNCTION
        void integrate_velocities(
            double *velocities,
            double *forces,
            double  mass
        )
        {
            const auto dt             = _dt.d_view();
            const auto velocityFactor = _velocityFactor.d_view();

            velocities[0] += dt * forces[0] / mass * velocityFactor;
            velocities[1] += dt * forces[1] / mass * velocityFactor;
            velocities[2] += dt * forces[2] / mass * velocityFactor;
        }
    };
}   // namespace integrator

#endif   // _KOKKOS_INTEGRATOR_HPP_