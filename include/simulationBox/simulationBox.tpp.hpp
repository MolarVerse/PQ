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

#ifndef _SIMULATION_BOX_TPP_

#define _SIMULATION_BOX_TPP_

#include "simulationBox.hpp"

namespace simulationBox
{
    /**
     * @brief set the box depending on dynamic type
     *
     * @tparam T
     * @param box
     */
    template <typename T>
    void SimulationBox::setBox(const T& box)
    {
        _box = std::make_shared<T>(box);
    }

}   // namespace simulationBox

#endif   // _SIMULATION_BOX_TPP_