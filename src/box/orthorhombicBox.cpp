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

#include "orthorhombicBox.hpp"

#include "constants.hpp"   // for _KG_PER_LITER_TO_AMU_PER_ANGSTROM_CUBIC_

using simulationBox::OrthorhombicBox;

/**
 * @brief Calculate the volume of the box
 *
 * @return volume
 */
double OrthorhombicBox::calculateVolume()
{
    _volume = _boxDimensions[0] * _boxDimensions[1] * _boxDimensions[2];

    return _volume;
}

/**
 * @brief applies the periodic boundary conditions
 *
 * @param position
 */
void OrthorhombicBox::applyPBC(linearAlgebra::Vec3D &position) const
{
    position -= _boxDimensions * round(position / _boxDimensions);
}

/**
 * @brief Calculate the shift vector
 *
 * @param shiftVector
 * @return linearAlgebra::Vec3D
 */
linearAlgebra::Vec3D OrthorhombicBox::calculateShiftVector(const linearAlgebra::Vec3D &shiftVector) const
{
    return _boxDimensions * round(shiftVector / _boxDimensions);
}

/**
 * @brief Calculate the box dimensions from the density
 *
 * @return vector<double>
 */
linearAlgebra::Vec3D OrthorhombicBox::calculateBoxDimensionsFromDensity(const double totalMass, const double density)
{
    _volume = totalMass / (density * constants::_KG_PER_LITER_TO_AMU_PER_ANGSTROM_CUBIC_);

    return linearAlgebra::Vec3D(::cbrt(_volume));
}