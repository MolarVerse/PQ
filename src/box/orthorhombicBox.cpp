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

#include "orthorhombicBox.hpp"

#include "constants.hpp"   // for _KG_PER_LITER_TO_AMU_PER_ANGSTROM_CUBIC_

using simulationBox::OrthorhombicBox;
using namespace linearAlgebra;
using namespace constants;

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
 * @brief Check if the box is orthorhombic
 *
 * @return true
 * @return false
 */
bool OrthorhombicBox::isOrthoRhombic() const { return true; }

/**
 * @brief applies the periodic boundary conditions
 *
 * @param position
 */
void OrthorhombicBox::applyPBC(Vec3D& position) const
{
    position -= _boxDimensions * round(position / _boxDimensions);
}

/**
 * @brief Calculate the shift vector
 *
 * @param shiftVector
 * @return Vec3D
 */
Vec3D OrthorhombicBox::calcShiftVector(const Vec3D& shiftVector) const
{
    return _boxDimensions * round(shiftVector / _boxDimensions);
}

/**
 * @brief Calculate the box dimensions from the density
 *
 * @return vector<double>
 */
Vec3D OrthorhombicBox::calcBoxDimFromDensity(
    const double totalMass,
    const double density
)
{
    _volume = totalMass / (density * _KG_PER_L_TO_AMU_PER_ANGSTROM3_);

    return Vec3D(::cbrt(_volume));
}

/**
 * @brief scales the cell dimensions and recalculates the volume
 *
 * @param scalingFactors
 */
void OrthorhombicBox::scaleBox(const tensor3D& scalingTensor)
{
    setBoxDimensions(_boxDimensions *= diagonal(scalingTensor));
    _volume = calculateVolume();
}