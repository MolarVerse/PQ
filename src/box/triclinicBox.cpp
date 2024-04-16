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

#include "triclinicBox.hpp"

#include "constants.hpp"          // for constants
#include "manostatSettings.hpp"   // for ManostatSettings

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
 * @details convert the angles from degrees to radians, calculate the transformation matrix and the box matrix
 *
 * @param boxAngles
 */
void TriclinicBox::setBoxAngles(const linearAlgebra::Vec3D &boxAngles)
{
    _boxAngles = boxAngles * constants::_DEG_TO_RAD_;

    calculateTransformationMatrix();
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
    _boxMatrix[0][1] = _boxDimensions[1] * _transformationMatrix[0][1];
    _boxMatrix[0][2] = _boxDimensions[2] * _transformationMatrix[0][2];

    _boxMatrix[1][1] = _boxDimensions[1] * _transformationMatrix[1][1];
    _boxMatrix[1][2] = _boxDimensions[2] * _transformationMatrix[1][2];

    _boxMatrix[2][2] = _boxDimensions[2] * _transformationMatrix[2][2];
}

/**
 * @brief Calculate the rotation matrix
 *
 */
void TriclinicBox::calculateTransformationMatrix()
{
    _transformationMatrix[0][0] = 1.0;
    _transformationMatrix[0][1] = cosGamma();
    _transformationMatrix[0][2] = cosBeta();

    _transformationMatrix[1][1] = sinGamma();
    _transformationMatrix[1][2] = (cosAlpha() - cosBeta() * cosGamma()) / sinGamma();

    _transformationMatrix[2][2] = ::sqrt(1.0 - sum(cos(_boxAngles) * cos(_boxAngles)) + 2 * prod(cos(_boxAngles))) / sinGamma();
}

/**
 * @brief applies the periodic boundary conditions
 *
 * @param position
 */
void TriclinicBox::applyPBC(linearAlgebra::Vec3D &position) const
{
    auto fractionalPosition = inverse(_boxMatrix) * position;

    fractionalPosition -= round(fractionalPosition);

    position = _boxMatrix * fractionalPosition;
}

/**
 * @brief Calculate the shift vector
 *
 * @param shiftVector
 * @return linearAlgebra::Vec3D
 */
linearAlgebra::Vec3D TriclinicBox::calculateShiftVector(const linearAlgebra::Vec3D &shiftVector) const
{
    return _boxMatrix * round(inverse(_boxMatrix) * shiftVector);
}

/**
 * @brief transform a vector into the orthogonal space
 *
 * @param vec
 * @return linearAlgebra::Vec3D
 */
linearAlgebra::Vec3D TriclinicBox::transformIntoOrthogonalSpace(const linearAlgebra::Vec3D &vec) const
{
    return inverse(_transformationMatrix) * vec;
}

/**
 * @brief transform a matrix into the orthogonal space
 *
 * @param mat
 * @return linearAlgebra::tensor3D
 */
linearAlgebra::tensor3D TriclinicBox::transformIntoOrthogonalSpace(const linearAlgebra::tensor3D &mat) const
{
    return inverse(_transformationMatrix) * mat;
}

/**
 * @brief transform a vector into the simulation space
 *
 * @param vec
 * @return linearAlgebra::Vec3D
 */
linearAlgebra::Vec3D TriclinicBox::transformIntoSimulationSpace(const linearAlgebra::Vec3D &vec) const
{
    return _transformationMatrix * vec;
}

/**
 * @brief transform a matrix into the simulation space
 *
 * @param mat
 * @return linearAlgebra::tensor3D
 */
linearAlgebra::tensor3D TriclinicBox::transformIntoSimulationSpace(const linearAlgebra::tensor3D &mat) const
{
    return _transformationMatrix * mat;
}

/**
 * @brief scale box dimensions and angles and recalculate the box matrix, transformation matrix and volume
 *
 * @details it first calculates the new box matrix, then the new box dimensions and angles. By setting the box dimensions and
 * angles the transformation matrix and volume are recalculated
 *
 * @param scalingTensor
 */
void TriclinicBox::scaleBox(const linearAlgebra::tensor3D &scalingTensor)
{
    if (settings ::ManostatSettings::getIsotropy() != settings::Isotropy::FULL_ANISOTROPIC)
        setBoxDimensions(diagonal(scalingTensor) * _boxDimensions);
    else
    {
        const auto boxMatrix = scalingTensor * _boxMatrix;

        const auto &[boxDimensions, boxAngles] = calculateBoxDimensionsAndAnglesFromBoxMatrix(boxMatrix);

        setBoxDimensions(boxDimensions);
        setBoxAngles(boxAngles);
    }

    _volume = calculateVolume();
}

/**
 * @brief determine box dimensions and angles from box matrix
 *
 * @param boxMatrix
 * @return std::pair<linearAlgebra::Vec3D, linearAlgebra::Vec3D>
 */
std::pair<linearAlgebra::Vec3D, linearAlgebra::Vec3D>
simulationBox::calculateBoxDimensionsAndAnglesFromBoxMatrix(const linearAlgebra::tensor3D &boxMatrix)
{
    const auto box_x = boxMatrix[0][0];
    const auto box_y = ::sqrt(boxMatrix[1][1] * boxMatrix[1][1] + boxMatrix[0][1] * boxMatrix[0][1]);
    const auto box_z =
        ::sqrt(boxMatrix[2][2] * boxMatrix[2][2] + boxMatrix[1][2] * boxMatrix[1][2] + boxMatrix[0][2] * boxMatrix[0][2]);

    const auto cos_alpha = (boxMatrix[0][1] * boxMatrix[0][2] + boxMatrix[1][1] * boxMatrix[1][2]) / (box_y * box_z);
    const auto cos_beta  = boxMatrix[0][2] / box_z;
    const auto cos_gamma = boxMatrix[0][1] / box_y;

    const auto alpha = ::acos(cos_alpha);
    const auto beta  = ::acos(cos_beta);
    const auto gamma = ::acos(cos_gamma);

    return std::make_pair(linearAlgebra::Vec3D{box_x, box_y, box_z},
                          linearAlgebra::Vec3D{alpha, beta, gamma} * constants::_RAD_TO_DEG_);
}