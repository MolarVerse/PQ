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
     * @class KokkosIntegrator
     *
     * @brief Integrator is a base class for all integrators
     *
     */
    class KokkosVelocityVerlet
    {
       private:
        Kokkos::DualView<double> _dt;
        Kokkos::DualView<double> _velocityFactor;
        Kokkos::DualView<double> _timeFactor;

       public:
        KOKKOS_FUNCTION
        void firstStep(
            simulationBox::SimulationBox       &simBox,
            simulationBox::KokkosSimulationBox &kokkosSimBox
        )
        {
            kokkosSimBox.transferPositionsFromSimulationBox(simBox);
            kokkosSimBox.transferForcesFromSimulationBox(simBox);
            kokkosSimBox.transferVelocitiesFromSimulationBox(simBox);
            kokkosSimBox.transferBoxDimensionsFromSimulationBox(simBox);

            auto       velocities    = kokkosSimBox.getVelocities().d_view;
            auto       forces        = kokkosSimBox.getForces().d_view;
            auto       masses        = kokkosSimBox.getMasses().d_view;
            auto       positions     = kokkosSimBox.getPositions().d_view;
            const auto boxDimensions = kokkosSimBox.getBoxDimensions().d_view;

            const auto dt             = _dt.d_view(0);
            const auto velocityFactor = _velocityFactor.d_view(0);
            const auto timeFactor     = _timeFactor.d_view(0);

            Kokkos::parallel_for(
                simBox.getNumberOfAtoms(),
                KOKKOS_LAMBDA(const size_t i) {
                    double pos[3] = {
                        positions(i, 0),
                        positions(i, 1),
                        positions(i, 2)
                    };

                    for (size_t j = 0; j < 3; ++j)
                    {
                        velocities(i, j) +=
                            dt * forces(i, j) / masses(i) * velocityFactor;

                        pos[j] += dt * velocities(i, j) * timeFactor;
                    }

                    double txyz[3] = {0.0, 0.0, 0.0};

                    simulationBox::KokkosSimulationBox::calculateShiftVector(
                        pos,
                        boxDimensions,
                        txyz
                    );

                    for (size_t j = 0; j < 3; ++j)
                    {
                        positions(i, j) += pos[j] + txyz[j];
                    }
                }
            );
        }
    };
}   // namespace integrator

#endif   // _KOKKOS_INTEGRATOR_HPP_