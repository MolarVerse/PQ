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

#ifndef _KOKKOS_SIMULATION_BOX_HPP_

#define _KOKKOS_SIMULATION_BOX_HPP_

#ifdef WITH_KOKKOS

#include <Kokkos_DualView.hpp>

/**
 * @namespace simulationBox
 */
namespace simulationBox
{
    /**
     * @class Kokkos SimulationBox
     *
     * @brief containing all information about the simulation box
     */
    class KokkosSimulationBox
    {
       private:
        Kokkos::DualView<size_t*> _atomTypes;
        Kokkos::DualView<size_t*> _molTypes;
        Kokkos::DualView<size_t*> _moleculeIndices;
        Kokkos::DualView<size_t*> _internalGlobalVDWTypes;

        Kokkos::DualView<double* [3], Kokkos::LayoutLeft> _positions;
        Kokkos::DualView<double* [3], Kokkos::LayoutLeft> _velocities;
        Kokkos::DualView<double* [3], Kokkos::LayoutLeft> _forces;
        Kokkos::DualView<double* [3], Kokkos::LayoutLeft> _shiftForces;
        Kokkos::DualView<double*>                         _partialCharges;
        Kokkos::DualView<double*>                         _masses;

        Kokkos::DualView<double*> _boxDimensions;

       public:
        explicit KokkosSimulationBox(const size_t numAtoms);

        KokkosSimulationBox()  = default;
        ~KokkosSimulationBox() = default;

        KOKKOS_FUNCTION static void calcShiftVector(
            const double*,
            Kokkos::View<double*>,
            double*
        );

        void initKokkosSimulationBox(simulationBox::SimulationBox& simBox);
        void initForces();

        void transferAtomTypesFromSimulationBox(
            simulationBox::SimulationBox& simBox
        );
        void transferMolTypesFromSimulationBox(
            simulationBox::SimulationBox& simBox
        );
        void transferMoleculeIndicesFromSimulationBox(
            simulationBox::SimulationBox& simBox
        );
        void transferInternalGlobalVDWTypesFromSimulationBox(
            simulationBox::SimulationBox&
        );

        void transferPositionsFromSimulationBox(
            simulationBox::SimulationBox& simBox
        );
        void transferVelocitiesFromSimulationBox(
            simulationBox::SimulationBox& simBox
        );
        void transferForcesFromSimulationBox(
            simulationBox::SimulationBox& simBox
        );
        void transferPartialChargesFromSimulationBox(
            simulationBox::SimulationBox& simBox
        );
        void transferMassesFromSimulationBox(
            simulationBox::SimulationBox& simBox
        );
        void transferBoxDimensionsFromSimulationBox(
            const simulationBox::SimulationBox& simBox
        );

        void transferPositionsToSimulationBox(
            simulationBox::SimulationBox& simBox
        );
        void transferVelocitiesToSimulationBox(
            simulationBox::SimulationBox& simBox
        );
        void transferForcesToSimulationBox(
            simulationBox::SimulationBox& simBox
        );
        void transferShiftForcesToSimulationBox(
            simulationBox::SimulationBox& simBox
        );

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] Kokkos::DualView<size_t*>& getAtomTypes();
        [[nodiscard]] Kokkos::DualView<size_t*>& getMolTypes();
        [[nodiscard]] Kokkos::DualView<size_t*>& getMoleculeIndices();
        [[nodiscard]] Kokkos::DualView<size_t*>& getInternalGlobalVDWTypes();
        // clang-format off
        [[nodiscard]] Kokkos::DualView<double* [3], Kokkos::LayoutLeft>& getPositions();
        [[nodiscard]] Kokkos::DualView<double* [3], Kokkos::LayoutLeft>& getVelocities();
        [[nodiscard]] Kokkos::DualView<double* [3], Kokkos::LayoutLeft>& getForces();
        [[nodiscard]] Kokkos::DualView<double* [3], Kokkos::LayoutLeft>& getShiftForces();
        // clang-format on
        [[nodiscard]] Kokkos::DualView<double*>& getMasses();
        [[nodiscard]] Kokkos::DualView<double*>& getPartialCharges();
        [[nodiscard]] Kokkos::DualView<double*>  getBoxDimensions();
    };
}   // namespace simulationBox

#endif   // WITH_KOKKOS

#endif   // _KOKKOS_SIMULATION_BOX_HPP_
