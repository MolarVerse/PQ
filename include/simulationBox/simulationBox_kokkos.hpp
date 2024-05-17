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

/**
 * @namespace Kokkos::simulationBox
 */
namespace Kokkos::simulationBox
{
    class SimulationBox
    {
       private:
        Kokkos::DualView<double*> _positions;
        Kokkos::DualView<double*> _velocities;
        Kokkos::DualView<double*> _forces;

       public:
        SimulationBox(size_t numAtoms);
        ~SimulationBox();
    };
}   // namespace Kokkos::simulationBox

#endif   // _KOKKOS_SIMULATION_BOX_HPP_