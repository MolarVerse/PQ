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

#ifndef MOLSYS__SIMULATION_BOX_TPP
#define MOLSYS__SIMULATION_BOX_TPP

#include "simulationBox.hpp"

namespace molsys
{
    /**
     * @brief set the box depending on dynamic type
     *
     * @tparam T
     * @param box
     * @param boxSetViaStartFile Indicates if the box was set via the start file
     */
    template <typename T>
    void SimulationBox::setBox(const T& box, bool boxSetViaStartFile)
    {
        _box                = std::make_shared<T>(box);
        _boxSetViaStartFile = boxSetViaStartFile;
    }

}   // namespace molsys

#endif   // MOLSYS__SIMULATION_BOX_TPP
