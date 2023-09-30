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

#include "triclinicBox.hpp"

#include "constants.hpp"   // for constants

using simulationBox::TriclinicBox;

/**
 * @brief Calculate the volume of the box
 *
 * @return volume
 */
double TriclinicBox::calculateVolume() { return det(_boxMatrix); }

/**
 * @brief set box angles and recalculate the box matrix
 *
 * @param boxAngles
 */
void TriclinicBox::setBoxAngles(const linearAlgebra::Vec3D &boxAngles)
{
    _boxAngles = boxAngles * constants::_RAD_TO_DEG_;

    calculateBoxMatrix();
}

/**
 * @brief set box dimensions and recalculate the box matrix
 *
 * @param boxDimensions
 */
void TriclinicBox::setBoxDimensions(const linearAlgebra::Vec3D &boxDimensions)
{
    _boxDimensions = boxDimensions;

    calculateBoxMatrix();
}

/**
 * @brief Calculate the box matrix from the box dimensions and angles
 *
 */
void TriclinicBox::calculateBoxMatrix()
{
    _boxMatrix[0][0] = _boxDimensions[0];
    _boxMatrix[0][1] = _boxDimensions[1] * cosGamma();
    _boxMatrix[0][2] = _boxDimensions[2] * cosBeta();

    _boxMatrix[1][1] = _boxDimensions[1] * sinGamma();
    _boxMatrix[1][2] = (_boxDimensions[2] * (cosAlpha() - cosBeta() * cosGamma())) / sinGamma();

    _boxMatrix[2][2] = ::sqrt(1.0 - sum(cos(_boxAngles) * cos(_boxAngles)) - 2 * prod(cos(_boxAngles))) / sinGamma();
}