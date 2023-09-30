/*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
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

#include "box.hpp"

#include "constants/conversionFactors.hpp"   // for _KG_PER_LITER_TO_AMU_PER_ANGSTROM_CUBIC_

#include <cmath>   // for cos, M_PI, cbrt, sqrt

using namespace simulationBox;

/**
 * @brief scales the cell dimensions and recalculates the volume
 *
 * @param scalingFactors
 */
void Box::scaleBox(const linearAlgebra::Vec3D &scalingFactors)
{
    _boxDimensions *= scalingFactors;
    calculateVolume();
}