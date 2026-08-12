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

#include "timerId.hpp"

/**
 * @brief convert TimerId to string
 *
 * @param id
 * @return std::string
 */
std::string toString(TimerId id)
{
    if (id == TimerId::CellList)
        return "Cell List";

    if (id == TimerId::PhysicalData)
        return "Physical Data";

    if (id == TimerId::ResetKinetics)
        return "Reset Kinetics";

    if (id == TimerId::WaterIntraPotential)
        return "Water Intra Potential";

    if (id == TimerId::WaterInterPotential)
        return "Water Inter Potential";

    if (id == TimerId::QMEngine)
        return "QM Engine";

    return TimerIdMeta::toString(id);
}
