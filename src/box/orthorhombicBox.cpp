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

/**
 * @file orthorhombicBox.cpp
 * @author Jakob Gamper (97gamjak@gmail.com)
 * @brief This file contains the implementation of the orthorhombic box class.
 * The orthorhombic box class is a derived class of the box class. It contains
 * the methods which are needed for the orthorhombic box. The orthorhombic box
 * is a box with right angles.
 *
 * @date 2024-12-09
 *
 * @see orthorhombicBox.hpp
 * @see box.hpp
 */

#include "orthorhombicBox.hpp"

#include "constants.hpp"   // for _KG_PER_LITER_TO_AMU_PER_ANGSTROM_CUBIC_
#include "settings.hpp"    // for Settings

using simulationBox::OrthorhombicBox;
using namespace linearAlgebra;
using namespace constants;
using namespace settings;

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

/**
 * @brief update box parameters
 *
 * @TODO: remove this later on should not be necessary
 *
 */
void OrthorhombicBox::flattenBoxParams()
{
    _boxParams = {_boxDimensions[0], _boxDimensions[1], _boxDimensions[2]};

#ifdef __PQ_GPU__
    if (Settings::useDevice())
    {
        copyBoxParamsTo();
    }
#endif
}

/**
 * @brief de-flatten box parameters
 *
 *
 */
void OrthorhombicBox::deFlattenBoxParams()
{
#ifdef __PQ_GPU__
    if (Settings::useDevice())
    {
        copyBoxParamsFrom();
    }
#endif

    _boxDimensions = {_boxParams[0], _boxParams[1], _boxParams[2]};
}
