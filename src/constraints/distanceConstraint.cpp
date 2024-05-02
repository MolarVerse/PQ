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

#include "distanceConstraint.hpp"

#include "simulationBox.hpp"

#include <cmath>
#include <vector>

using namespace constraints;

/**
 * @brief calculates the reference distance of all distance constraints
 *
 * @param simulationBox
 *
 */
void DistanceConstraint::applyDistanceConstraint(const simulationBox::SimulationBox &simulationBox)
{
    const auto pos1 = _molecules[0]->getAtomPosition(_atomIndices[0]);
    const auto pos2 = _molecules[1]->getAtomPosition(_atomIndices[1]);

    auto dPos = pos1 - pos2;
    simulationBox.applyPBC(dPos);
}