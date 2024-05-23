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

#include "integrator_kokkos.hpp"

using namespace integrator;

/**
 * @brief constructor
 *
 * @param dt             time step
 * @param velocityFactor velocity factor
 * @param timeFactor     time factor
 */
KokkosVelocityVerlet::KokkosVelocityVerlet(
    const double dt,
    const double velocityFactor,
    const double timeFactor
)
    : _dt("dt", 1),
      _velocityFactor("velocityFactor", 1),
      _timeFactor("timeFactor", 1)
{
    _dt.h_view()             = dt;
    _velocityFactor.h_view() = velocityFactor;
    _timeFactor.h_view()     = timeFactor;

    Kokkos::deep_copy(_dt.d_view, _dt.h_view);
    Kokkos::deep_copy(_velocityFactor.d_view, _velocityFactor.h_view);
    Kokkos::deep_copy(_timeFactor.d_view, _timeFactor.h_view);
}

/**
 * @brief first step of the velocity Verlet integrator
 *
 * @param simBox      simulation box
 * @param kokkosSimBox Kokkos simulation box
 */
void KokkosVelocityVerlet::firstStep(
    simulationBox::SimulationBox       &simBox,
    simulationBox::KokkosSimulationBox &kokkosSimBox
)
{
    startTimingsSection("Velocity Verlet - first step");

    kokkosSimBox.transferPositionsFromSimulationBox(simBox);
    kokkosSimBox.transferForcesFromSimulationBox(simBox);
    kokkosSimBox.transferVelocitiesFromSimulationBox(simBox);
    kokkosSimBox.transferBoxDimensionsFromSimulationBox(simBox);

    auto       velocities    = kokkosSimBox.getVelocities().d_view;
    auto       forces        = kokkosSimBox.getForces().d_view;
    auto       masses        = kokkosSimBox.getMasses().d_view;
    auto       positions     = kokkosSimBox.getPositions().d_view;
    const auto boxDimensions = kokkosSimBox.getBoxDimensions().d_view;

    const auto dt             = _dt.d_view;
    const auto timeFactor     = _timeFactor.d_view;
    const auto velocityFactor = _velocityFactor.d_view;

    Kokkos::parallel_for(
        simBox.getNumberOfAtoms(),
        KOKKOS_LAMBDA(const size_t i) {
            float pos[3] = {positions(i, 0), positions(i, 1), positions(i, 2)};

            // integrate_velocities(&velocities(i, 0), &forces(i, 0),
            // masses(i));

            for (size_t j = 0; j < 3; ++j)
            {
                velocities(i, j) +=
                    forces(i, j) / masses(i) * dt() * velocityFactor();

                pos[j] += dt() * velocities(i, j) * timeFactor();
            }

            float txyz[3] = {0.0, 0.0, 0.0};

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
    );   // TODO: calculate new center of mass

    kokkosSimBox.transferVelocitiesToSimulationBox(simBox);
    kokkosSimBox.transferPositionsToSimulationBox(simBox);

    stopTimingsSection("Velocity Verlet - first step");
}

/**
 * @brief second step of the velocity Verlet integrator
 *
 * @param simBox      simulation box
 * @param kokkosSimBox Kokkos simulation box
 */
void KokkosVelocityVerlet::secondStep(
    simulationBox::SimulationBox       &simBox,
    simulationBox::KokkosSimulationBox &kokkosSimBox
)
{
    startTimingsSection("Velocity Verlet - second step");

    kokkosSimBox.transferForcesFromSimulationBox(simBox);
    kokkosSimBox.transferVelocitiesFromSimulationBox(simBox);

    auto forces     = kokkosSimBox.getForces().d_view;
    auto velocities = kokkosSimBox.getVelocities().d_view;
    auto masses     = kokkosSimBox.getMasses().d_view;

    const auto dt             = _dt.d_view;
    const auto velocityFactor = _velocityFactor.d_view;

    Kokkos::parallel_for(
        simBox.getNumberOfAtoms(),
        KOKKOS_LAMBDA(const size_t i) {
            // integrate_velocities(&velocities(i, 0), &forces(i, 0),
            // masses(i));

            for (size_t j = 0; j < 3; ++j)
            {
                velocities(i, j) +=
                    forces(i, j) / masses(i) * dt() * velocityFactor();
            }
        }
    );

    kokkosSimBox.transferVelocitiesToSimulationBox(simBox);

    stopTimingsSection("Velocity Verlet - second step");
}