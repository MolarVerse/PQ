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

#include <Kokkos_DualView.hpp>

#include "simulationBox.hpp"   // for SimulationBox
#include "vector3d.hpp"        // for Vector3D

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
        Kokkos::DualView<size_t*> _internalGlobalVDWTypes;

        Kokkos::DualView<double* [3]> _positions;
        Kokkos::DualView<double* [3]> _forces;
        Kokkos::DualView<double* [3]> _shiftForces;
        Kokkos::DualView<double*>     _partialCharges;

        Kokkos::DualView<double[3]> _boxDimensions =
            Kokkos::DualView<double[3]>("boxDimensions");

       public:
        KokkosSimulationBox(size_t numAtoms);

        KokkosSimulationBox()  = default;
        ~KokkosSimulationBox() = default;

        KOKKOS_FUNCTION static void calculateShiftVector(
            const double* dxyz,
            const double* boxDimensions,
            double*       txyz
        );

        void transferAtomTypesFromSimulationBox(SimulationBox& simBox);
        void transferMolTypesFromSimulationBox(SimulationBox& simBox);
        void transferInternalGlobalVDWTypesFromSimulationBox(
            SimulationBox& simBox
        );

        void transferPositionsFromSimulationBox(SimulationBox& simBox);
        void transferPartialChargesFromSimulationBox(SimulationBox& simBox);
        void transferBoxDimensionsFromSimulationBox(SimulationBox& simBox);

        void initializeForces();
        void initializeShiftForces();
        void transferForcesToSimulationBox(SimulationBox& simBox);
        void transferShiftForcesToSimulationBox(SimulationBox& simBox);

        // getters
        [[nodiscard]] Kokkos::DualView<size_t*>& getAtomTypes()
        {
            return _atomTypes;
        }
        [[nodiscard]] Kokkos::DualView<size_t*>& getMolTypes()
        {
            return _molTypes;
        }
        [[nodiscard]] Kokkos::DualView<size_t*>& getInternalGlobalVDWTypes()
        {
            return _internalGlobalVDWTypes;
        }
        [[nodiscard]] Kokkos::DualView<double* [3]>& getPositions()
        {
            return _positions;
        }
        [[nodiscard]] Kokkos::DualView<double* [3]>& getForces()
        {
            return _forces;
        }
        [[nodiscard]] Kokkos::DualView<double* [3]>& getShiftForces()
        {
            return _shiftForces;
        }
        [[nodiscard]] Kokkos::DualView<double*>& getPartialCharges()
        {
            return _partialCharges;
        }
        [[nodiscard]] Kokkos::DualView<double[3]>& getBoxDimensions()
        {
            return _boxDimensions;
        }
    };
}   // namespace simulationBox

#endif   // _KOKKOS_SIMULATION_BOX_HPP_