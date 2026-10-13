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

#include "box.hpp"
#include "constants.hpp"
#include "manostatSettings.hpp"
#include "staticMatrix.hpp"
#include "vector3d.hpp"

namespace molsys
{

    /**
     * @brief Construct a new Triclinic Box object
     *
     * @param boxDimensions
     * @param boxAngles
     */
    TriclinicBox::TriclinicBox(
        const linalg::Vec3D &boxDimensions,
        const linalg::Vec3D &boxAngles
    )
        : Box(boxDimensions, boxAngles * DEG_TO_RAD)
    {
    }

    /**
     * @brief Calculate the volume of the box
     *
     * @return volume
     */
    double TriclinicBox::_calculateVolume(const linalg::tensor3D &boxMatrix)
    {
        return det(boxMatrix);
    }

    /**
     * @brief Get the volume of the box (cached)
     *
     * @return volume
     */
    double TriclinicBox::getVolume() const { return _volume.get(*this); }

    /**
     * @brief Calculate the box matrix from the box dimensions and angles
     *
     */
    linalg::tensor3D TriclinicBox::_calculateBoxMatrix(
        const linalg::Vec3D    &boxDimensions,
        const linalg::tensor3D &transformationMatrix
    )
    {
        linalg::tensor3D boxMatrix{0.0};

        boxMatrix[0] = boxDimensions * transformationMatrix[0];
        boxMatrix[1] = boxDimensions * transformationMatrix[1];
        boxMatrix[2] = boxDimensions * transformationMatrix[2];

        return boxMatrix;
    }

    /**
     * @brief Calculate the rotation matrix
     *
     */
    linalg::tensor3D TriclinicBox::_calculateTransformationMatrix(
        const linalg::Vec3D &boxAngles
    )
    {
        const auto cosGamma = _cosGamma(boxAngles);
        const auto sinGamma = _sinGamma(boxAngles);
        const auto cosBeta  = _cosBeta(boxAngles);

        linalg::tensor3D transformationMatrix{0.0};

        transformationMatrix[0][0] = 1;
        transformationMatrix[0][1] = cosGamma;
        transformationMatrix[0][2] = cosBeta;

        transformationMatrix[1][1]  = sinGamma;
        transformationMatrix[1][2]  = _cosAlpha(boxAngles) - cosBeta * cosGamma;
        transformationMatrix[1][2] /= sinGamma;

        const auto cosAngles = cos(boxAngles);
        const auto sumcos_2  = sum(cosAngles * cosAngles);
        const auto prodcos   = prod(cosAngles);

        transformationMatrix[2][2]  = ::sqrt(1 - sumcos_2 + (2 * prodcos));
        transformationMatrix[2][2] /= sinGamma;

        return transformationMatrix;
    }

    /**
     * @brief Calculate the inverse of the box matrix
     *
     */
    linalg::tensor3D TriclinicBox::_calculateInverse(
        const linalg::tensor3D &boxMatrix
    )
    {
        return inverse(boxMatrix);
    }

    /**
     * @brief applies the periodic boundary conditions
     *
     * @param position
     */
    void TriclinicBox::applyPBC(linalg::Vec3D &position) const
    {
        const auto originalPosition = position;

        auto fractionalPosition = _inverse.get(*this) * position;

        fractionalPosition -= round(fractionalPosition);

        const auto boxMatrix = _boxMatrix.get(*this);
        position             = boxMatrix * fractionalPosition;

        const auto distance = norm(position);

        linalg::Vec3D analyticPosition   = position;
        double        analyticalDistance = distance;

        const auto minimalBoxDimensionHalf = getMinimalBoxDimension() / 2.0;

        if (distance > minimalBoxDimensionHalf)
        {
            for (int i = -1; i <= 1; ++i)
            {
                for (int j = -1; j <= 1; ++j)
                {
                    for (int k = -1; k <= 1; ++k)
                    {
                        const auto shift =
                            boxMatrix * linalg::Vec3D{
                                            static_cast<double>(i),
                                            static_cast<double>(j),
                                            static_cast<double>(k),
                                        };

                        const auto newPosition = originalPosition + shift;

                        const auto newDistance = norm(newPosition);

                        if (newDistance < analyticalDistance)
                        {
                            analyticPosition   = newPosition;
                            analyticalDistance = newDistance;
                        }
                    }
                }
            }
        }

        position = analyticPosition;
    }

    /**
     * @brief wraps atoms into the primary triclinic box unit cell
     *
     * @param pos position to be wrapped
     *
     * @details This function images atoms back into the triclinic box with
     * fractional coordinates in the range [-0.5, 0.5). This centers the box
     * at the origin and is useful for output visualization where you want all
     * atoms within the primary unit cell rather than the nearest image to the
     * origin.
     *
     * @return linalg::Vec3D wrapped position with box centered at origin
     */
    linalg::Vec3D TriclinicBox::wrapPositionIntoBox(
        const linalg::Vec3D &pos
    ) const
    {
        auto fractionalPosition = _inverse.get(*this) * pos;

        constexpr auto shiftFraction = 0.5;

        fractionalPosition -= floor(fractionalPosition + shiftFraction);

        return _boxMatrix.get(*this) * fractionalPosition;
    }

    /**
     * @brief Calculate the shift vector
     *
     * @param vec
     * @return linalg::Vec3D
     */
    linalg::Vec3D TriclinicBox::calcShiftVector(const linalg::Vec3D &vec) const
    {
        return _boxMatrix.get(*this) * round(_inverse.get(*this) * vec);
    }

    /**
     * @brief transform a vector into the orthogonal space
     *
     * @param vec
     * @return linalg::Vec3D
     */
    linalg::Vec3D TriclinicBox::toOrthoSpace(const linalg::Vec3D &vec) const
    {
        return _invTrafoMatrix.get(*this) * vec;
    }

    /**
     * @brief transform a matrix into the orthogonal space
     *
     * @param mat
     * @return linalg::tensor3D
     */
    linalg::tensor3D TriclinicBox::toOrthoSpace(
        const linalg::tensor3D &mat
    ) const
    {
        return _invTrafoMatrix.get(*this) * mat;
    }

    /**
     * @brief transform a vector into the simulation space
     *
     * @param vec
     * @return linalg::Vec3D
     */
    linalg::Vec3D TriclinicBox::toSimSpace(const linalg::Vec3D &vec) const
    {
        return _trafoMatrix.get(*this) * vec;
    }

    /**
     * @brief transform a matrix into the simulation space
     *
     * @param mat
     * @return linalg::tensor3D
     */
    linalg::tensor3D TriclinicBox::toSimSpace(const linalg::tensor3D &mat) const
    {
        return _trafoMatrix.get(*this) * mat;
    }

    /**
     * @brief scale box dimensions and angles and recalculate the box matrix,
     * transformation matrix and volume
     *
     * @details it first calculates the new box matrix, then the new box
     * dimensions and angles. By setting the box dimensions and angles the
     * transformation matrix and volume are recalculated
     *
     * @param scalingTensor
     */
    void TriclinicBox::scaleBox(const linalg::tensor3D &scalingTensor)
    {
        if (settings::ManostatSettings::getIsotropy() !=
            Isotropy::FULL_ANISOTROPIC)
        {
            _boxDimensions.set(diagonal(scalingTensor) * _boxDimensions.get());
        }
        else
        {
            const auto boxMatrix = scalingTensor * _boxMatrix.get(*this);

            const auto &[boxDimensions, boxAngles] =
                calcBoxDimAndAnglesFromBoxMatrix(boxMatrix);

            _boxDimensions.set(boxDimensions);
            _angles.set(boxAngles);
        }
    }

    /**
     * @brief determine box dimensions and angles from box matrix (angles are in
     * RAD)
     *
     * @param boxMatrix
     * @return std::pair<linalg::Vec3D, linalg::Vec3D>
     */
    std::pair<linalg::Vec3D, linalg::Vec3D> calcBoxDimAndAnglesFromBoxMatrix(
        const linalg::tensor3D &boxMatrix
    )
    {
        const auto box_x = boxMatrix[0][0];
        const auto box_y = ::sqrt(
            (boxMatrix[1][1] * boxMatrix[1][1]) +
            (boxMatrix[0][1] * boxMatrix[0][1])
        );
        const auto box_z = ::sqrt(
            (boxMatrix[2][2] * boxMatrix[2][2]) +
            (boxMatrix[1][2] * boxMatrix[1][2]) +
            (boxMatrix[0][2] * boxMatrix[0][2])
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
            linalg::Vec3D{box_x, box_y, box_z},
            linalg::Vec3D{alpha, beta, gamma}
        );
    }

    /**
     * @brief get the minimal box dimension
     *
     * @return double
     */
    double TriclinicBox::getMinimalBoxDimension() const
    {
        return minimum(diagonal(_boxMatrix.get(*this)));
    }

    /**
     * @brief calculate cos of alpha
     *
     * @return double
     */
    double TriclinicBox::_cosAlpha(const linalg::Vec3D &boxAngles)
    {
        return ::cos(boxAngles[0]);
    }

    /**
     * @brief calculate cos of beta
     *
     * @return double
     */
    double TriclinicBox::_cosBeta(const linalg::Vec3D &boxAngles)
    {
        return ::cos(boxAngles[1]);
    }

    /**
     * @brief calculate cos of gamma
     *
     * @return double
     */
    double TriclinicBox::_cosGamma(const linalg::Vec3D &boxAngles)
    {
        return ::cos(boxAngles[2]);
    }

    /**
     * @brief calculate sin of alpha
     *
     * @return double
     */
    double TriclinicBox::_sinAlpha(const linalg::Vec3D &boxAngles)
    {
        return ::sin(boxAngles[0]);
    }

    /**
     * @brief calculate sin of beta
     *
     * @return double
     */
    double TriclinicBox::_sinBeta(const linalg::Vec3D &boxAngles)
    {
        return ::sin(boxAngles[1]);
    }

    /**
     * @brief calculate sin of gamma
     *
     * @return double
     */
    double TriclinicBox::_sinGamma(const linalg::Vec3D &boxAngles)
    {
        return ::sin(boxAngles[2]);
    }

    /**
     * @brief get the box matrix
     *
     * @return linalg::tensor3D
     */
    linalg::tensor3D TriclinicBox::getBoxMatrix(Periodicity per) const
    {
        auto boxMatrix = _boxMatrix.get(*this);

        switch (per)
        {
            case Periodicity::NON_PERIODIC:
                boxMatrix[0][0] =
                    defaults::VACUUM_BOX_DIMENSION;   // X dimension
                boxMatrix[1][1] =
                    defaults::VACUUM_BOX_DIMENSION;   // Y dimension
                boxMatrix[2][2] =
                    defaults::VACUUM_BOX_DIMENSION;   // Z dimension
                boxMatrix[0][1] = 0.0;                // Clear XY cross term
                boxMatrix[0][2] = 0.0;                // Clear XZ cross term
                boxMatrix[1][0] = 0.0;                // Clear YX cross term
                boxMatrix[1][2] = 0.0;                // Clear YZ cross term
                boxMatrix[2][0] = 0.0;                // Clear ZX cross term
                boxMatrix[2][1] = 0.0;                // Clear ZY cross term
                break;
            case Periodicity::X:
                boxMatrix[1][1] =
                    defaults::VACUUM_BOX_DIMENSION;   // Y dimension
                boxMatrix[2][2] =
                    defaults::VACUUM_BOX_DIMENSION;   // Z dimension
                boxMatrix[0][1] = 0.0;                // Clear XY cross term
                boxMatrix[0][2] = 0.0;                // Clear XZ cross term
                boxMatrix[1][0] = 0.0;                // Clear YX cross term
                boxMatrix[1][2] = 0.0;                // Clear YZ cross term
                boxMatrix[2][0] = 0.0;                // Clear ZX cross term
                boxMatrix[2][1] = 0.0;                // Clear ZY cross term
                break;
            case Periodicity::Y:
                boxMatrix[0][0] =
                    defaults::VACUUM_BOX_DIMENSION;   // X dimension
                boxMatrix[2][2] =
                    defaults::VACUUM_BOX_DIMENSION;   // Z dimension
                boxMatrix[0][1] = 0.0;                // Clear XY cross term
                boxMatrix[0][2] = 0.0;                // Clear XZ cross term
                boxMatrix[1][0] = 0.0;                // Clear YX cross term
                boxMatrix[1][2] = 0.0;                // Clear YZ cross term
                boxMatrix[2][0] = 0.0;                // Clear ZX cross term
                boxMatrix[2][1] = 0.0;                // Clear ZY cross term
                break;
            case Periodicity::Z:
                boxMatrix[0][0] =
                    defaults::VACUUM_BOX_DIMENSION;   // X dimension
                boxMatrix[1][1] =
                    defaults::VACUUM_BOX_DIMENSION;   // Y dimension
                boxMatrix[0][1] = 0.0;                // Clear XY cross term
                boxMatrix[0][2] = 0.0;                // Clear XZ cross term
                boxMatrix[1][0] = 0.0;                // Clear YX cross term
                boxMatrix[1][2] = 0.0;                // Clear YZ cross term
                boxMatrix[2][0] = 0.0;                // Clear ZX cross term
                boxMatrix[2][1] = 0.0;                // Clear ZY cross term
                break;
            case Periodicity::XY:
                boxMatrix[2][2] =
                    defaults::VACUUM_BOX_DIMENSION;   // Z dimension
                boxMatrix[0][2] = 0.0;                // Clear XZ cross term
                boxMatrix[1][2] = 0.0;                // Clear YZ cross term
                boxMatrix[2][0] = 0.0;                // Clear ZX cross term
                boxMatrix[2][1] = 0.0;                // Clear ZY cross term
                break;
            case Periodicity::XZ:
                boxMatrix[1][1] =
                    defaults::VACUUM_BOX_DIMENSION;   // Y dimension
                boxMatrix[0][1] = 0.0;                // Clear XY cross term
                boxMatrix[1][0] = 0.0;                // Clear YX cross term
                boxMatrix[1][2] = 0.0;                // Clear YZ cross term
                boxMatrix[2][1] = 0.0;                // Clear ZY cross term
                break;
            case Periodicity::YZ:
                boxMatrix[0][0] =
                    defaults::VACUUM_BOX_DIMENSION;   // X dimension
                boxMatrix[0][1] = 0.0;                // Clear XY cross term
                boxMatrix[0][2] = 0.0;                // Clear XZ cross term
                boxMatrix[1][0] = 0.0;                // Clear YX cross term
                boxMatrix[2][0] = 0.0;                // Clear ZX cross term
                break;
            case Periodicity::XYZ: break;
        }

        return boxMatrix;
    }

}   // namespace molsys
