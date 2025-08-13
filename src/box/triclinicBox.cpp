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

using namespace linearAlgebra;
using namespace settings;
using namespace constants;

/**
 * @brief Calculate the volume of the box
 *
 * @return volume
 */
double TriclinicBox::calculateVolume() { return det(_boxMatrix); }

/**
 * @brief set box angles and recalculate the box matrix
 *
 * @details convert the angles from degrees to radians, calculate the
 * transformation matrix and the box matrix
 *
 * @param boxAngles
 */
void TriclinicBox::setBoxAngles(const Vec3D &boxAngles)
{
    _boxAngles = boxAngles * _DEG_TO_RAD_;

    calculateTransformationMatrix();
    calculateBoxMatrix();
}

/**
 * @brief set box dimensions and recalculate the box matrix
 *
 * @param boxDimensions
 */
void TriclinicBox::setBoxDimensions(const Vec3D &boxDimensions)
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

    _transformationMatrix[1][1]  = sinGamma();
    _transformationMatrix[1][2]  = cosAlpha() - cosBeta() * cosGamma();
    _transformationMatrix[1][2] /= sinGamma();

    const auto sumcos_2          = sum(cos(_boxAngles) * cos(_boxAngles));
    const auto prodcos           = prod(cos(_boxAngles));
    _transformationMatrix[2][2]  = ::sqrt(1.0 - sumcos_2 + 2 * prodcos);
    _transformationMatrix[2][2] /= sinGamma();
}

/**
 * @brief applies the periodic boundary conditions
 *
 * @param position
 */
void TriclinicBox::applyPBC(Vec3D &position) const
{
    const auto originalPosition = position;

    auto fractionalPosition = inverse(_boxMatrix) * position;

    fractionalPosition -= round(fractionalPosition);

    position = _boxMatrix * fractionalPosition;

    const auto distance = norm(position);

    Vec3D  analyticPosition   = position;
    double analyticalDistance = distance;

    if (distance > 0.5 * getMinimalBoxDimension())
    {
        for (int i = -1; i <= 1; ++i)
            for (int j = -1; j <= 1; ++j)
                for (int k = -1; k <= 1; ++k)
                {
                    const auto shift = _boxMatrix * Vec3D{i, j, k};

                    const auto newPosition = originalPosition + shift;

                    const auto newDistance = norm(newPosition);

                    if (newDistance < analyticalDistance)
                    {
                        analyticPosition   = newPosition;
                        analyticalDistance = newDistance;
                    }
                }
    }

    position = analyticPosition;
}

/**
 * @brief Calculate the shift vector
 *
 * @param shiftVector
 * @return Vec3D
 */
Vec3D TriclinicBox::calcShiftVector(const Vec3D &shiftVector) const
{
    return _boxMatrix * round(inverse(_boxMatrix) * shiftVector);
}

/**
 * @brief transform a vector into the orthogonal space
 *
 * @param vec
 * @return Vec3D
 */
Vec3D TriclinicBox::toOrthoSpace(const Vec3D &vec) const
{
    return inverse(_transformationMatrix) * vec;
}

/**
 * @brief transform a matrix into the orthogonal space
 *
 * @param mat
 * @return tensor3D
 */
tensor3D TriclinicBox::toOrthoSpace(const tensor3D &mat) const
{
    return inverse(_transformationMatrix) * mat;
}

/**
 * @brief transform a vector into the simulation space
 *
 * @param vec
 * @return Vec3D
 */
Vec3D TriclinicBox::toSimSpace(const Vec3D &vec) const
{
    return _transformationMatrix * vec;
}

/**
 * @brief transform a matrix into the simulation space
 *
 * @param mat
 * @return tensor3D
 */
tensor3D TriclinicBox::toSimSpace(const tensor3D &mat) const
{
    return _transformationMatrix * mat;
}

/**
 * @brief scale box dimensions and angles and recalculate the box matrix,
 * transformation matrix and volume
 *
 * @details it first calculates the new box matrix, then the new box dimensions
 * and angles. By setting the box dimensions and angles the transformation
 * matrix and volume are recalculated
 *
 * @param scalingTensor
 */
void TriclinicBox::scaleBox(const tensor3D &scalingTensor)
{
    if (ManostatSettings::getIsotropy() != Isotropy::FULL_ANISOTROPIC)
        setBoxDimensions(diagonal(scalingTensor) * _boxDimensions);

    else
    {
        const auto boxMatrix = scalingTensor * _boxMatrix;

        const auto &[boxDimensions, boxAngles] =
            calcBoxDimAndAnglesFromBoxMatrix(boxMatrix);

        setBoxDimensions(boxDimensions);
        setBoxAngles(boxAngles);
    }

    _volume = calculateVolume();
}

/**
 * @brief determine box dimensions and angles from box matrix
 *
 * @param boxMatrix
 * @return std::pair<Vec3D, Vec3D>
 */
std::pair<Vec3D, Vec3D> simulationBox::calcBoxDimAndAnglesFromBoxMatrix(
    const tensor3D &boxMatrix
)
{
    const auto box_x = boxMatrix[0][0];
    const auto box_y = ::sqrt(
        boxMatrix[1][1] * boxMatrix[1][1] + boxMatrix[0][1] * boxMatrix[0][1]
    );
    const auto box_z = ::sqrt(
        boxMatrix[2][2] * boxMatrix[2][2] + boxMatrix[1][2] * boxMatrix[1][2] +
        boxMatrix[0][2] * boxMatrix[0][2]
    );

    const auto cos_alpha = (boxMatrix[0][1] * boxMatrix[0][2] +
                            boxMatrix[1][1] * boxMatrix[1][2]) /
                           (box_y * box_z);

    const auto cos_beta  = boxMatrix[0][2] / box_z;
    const auto cos_gamma = boxMatrix[0][1] / box_y;

    const auto alpha = ::acos(cos_alpha);
    const auto beta  = ::acos(cos_beta);
    const auto gamma = ::acos(cos_gamma);

    return std::make_pair(
        Vec3D{box_x, box_y, box_z},
        Vec3D{alpha, beta, gamma} * constants::_RAD_TO_DEG_
    );
}

/**
 * @brief get the minimal box dimension
 *
 * @return double
 */
double TriclinicBox::getMinimalBoxDimension() const
{
    return minimum(diagonal(_boxMatrix));
}

/**
 * @brief calculate cos of alpha
 *
 * @return double
 */
double TriclinicBox::cosAlpha() const { return ::cos(_boxAngles[0]); }

/**
 * @brief calculate cos of beta
 *
 * @return double
 */
double TriclinicBox::cosBeta() const { return ::cos(_boxAngles[1]); }

/**
 * @brief calculate cos of gamma
 *
 * @return double
 */
double TriclinicBox::cosGamma() const { return ::cos(_boxAngles[2]); }

/**
 * @brief calculate sin of alpha
 *
 * @return double
 */
double TriclinicBox::sinAlpha() const { return ::sin(_boxAngles[0]); }

/**
 * @brief calculate sin of beta
 *
 * @return double
 */
double TriclinicBox::sinBeta() const { return ::sin(_boxAngles[1]); }

/**
 * @brief calculate sin of gamma
 *
 * @return double
 */
double TriclinicBox::sinGamma() const { return ::sin(_boxAngles[2]); }

/**
 * @brief get the box angles
 *
 * @return Vec3D
 */
Vec3D TriclinicBox::getBoxAngles() const
{
    return _boxAngles * constants::_RAD_TO_DEG_;
}

/**
 * @brief get the box matrix
 *
 * @return tensor3D
 */
tensor3D TriclinicBox::getBoxMatrix() const { return _boxMatrix; }

/**
 * @brief get the box matrix
 *
 * @return tensor3D
 */
tensor3D TriclinicBox::getBoxMatrix(Periodicity per) const
{
    using namespace defaults;

    auto boxMatrix = getBoxMatrix();

    switch (per)
    {
        case Periodicity::NON_PERIODIC:
            boxMatrix[0][0] = _VACUUM_BOX_DIMENSION_;   // X dimension
            boxMatrix[1][1] = _VACUUM_BOX_DIMENSION_;   // Y dimension
            boxMatrix[2][2] = _VACUUM_BOX_DIMENSION_;   // Z dimension
            boxMatrix[0][1] = 0.0;                      // Clear XY cross term
            boxMatrix[0][2] = 0.0;                      // Clear XZ cross term
            boxMatrix[1][0] = 0.0;                      // Clear YX cross term
            boxMatrix[1][2] = 0.0;                      // Clear YZ cross term
            boxMatrix[2][0] = 0.0;                      // Clear ZX cross term
            boxMatrix[2][1] = 0.0;                      // Clear ZY cross term
            break;
        case Periodicity::X:
            boxMatrix[1][1] = _VACUUM_BOX_DIMENSION_;   // Y dimension
            boxMatrix[2][2] = _VACUUM_BOX_DIMENSION_;   // Z dimension
            boxMatrix[0][1] = 0.0;                      // Clear XY cross term
            boxMatrix[0][2] = 0.0;                      // Clear XZ cross term
            boxMatrix[1][0] = 0.0;                      // Clear YX cross term
            boxMatrix[1][2] = 0.0;                      // Clear YZ cross term
            boxMatrix[2][0] = 0.0;                      // Clear ZX cross term
            boxMatrix[2][1] = 0.0;                      // Clear ZY cross term
            break;
        case Periodicity::Y:
            boxMatrix[0][0] = _VACUUM_BOX_DIMENSION_;   // X dimension
            boxMatrix[2][2] = _VACUUM_BOX_DIMENSION_;   // Z dimension
            boxMatrix[0][1] = 0.0;                      // Clear XY cross term
            boxMatrix[0][2] = 0.0;                      // Clear XZ cross term
            boxMatrix[1][0] = 0.0;                      // Clear YX cross term
            boxMatrix[1][2] = 0.0;                      // Clear YZ cross term
            boxMatrix[2][0] = 0.0;                      // Clear ZX cross term
            boxMatrix[2][1] = 0.0;                      // Clear ZY cross term
            break;
        case Periodicity::Z:
            boxMatrix[0][0] = _VACUUM_BOX_DIMENSION_;   // X dimension
            boxMatrix[1][1] = _VACUUM_BOX_DIMENSION_;   // Y dimension
            boxMatrix[0][1] = 0.0;                      // Clear XY cross term
            boxMatrix[0][2] = 0.0;                      // Clear XZ cross term
            boxMatrix[1][0] = 0.0;                      // Clear YX cross term
            boxMatrix[1][2] = 0.0;                      // Clear YZ cross term
            boxMatrix[2][0] = 0.0;                      // Clear ZX cross term
            boxMatrix[2][1] = 0.0;                      // Clear ZY cross term
            break;
        case Periodicity::XY:
            boxMatrix[2][2] = _VACUUM_BOX_DIMENSION_;   // Z dimension
            boxMatrix[0][2] = 0.0;                      // Clear XZ cross term
            boxMatrix[1][2] = 0.0;                      // Clear YZ cross term
            boxMatrix[2][0] = 0.0;                      // Clear ZX cross term
            boxMatrix[2][1] = 0.0;                      // Clear ZY cross term
            break;
        case Periodicity::XZ:
            boxMatrix[1][1] = _VACUUM_BOX_DIMENSION_;   // Y dimension
            boxMatrix[0][1] = 0.0;                      // Clear XY cross term
            boxMatrix[1][0] = 0.0;                      // Clear YX cross term
            boxMatrix[1][2] = 0.0;                      // Clear YZ cross term
            boxMatrix[2][1] = 0.0;                      // Clear ZY cross term
            break;
        case Periodicity::YZ:
            boxMatrix[0][0] = _VACUUM_BOX_DIMENSION_;   // X dimension
            boxMatrix[0][1] = 0.0;                      // Clear XY cross term
            boxMatrix[0][2] = 0.0;                      // Clear XZ cross term
            boxMatrix[1][0] = 0.0;                      // Clear YX cross term
            boxMatrix[2][0] = 0.0;                      // Clear ZX cross term
            break;
        // default also handles case Periodicity::XYZ
        default: break;
    }

    return boxMatrix;
}

/**
 * @brief get the transformation matrix
 *
 * @return tensor3D
 */
tensor3D TriclinicBox::getTransformationMatrix() const
{
    return _transformationMatrix;
}