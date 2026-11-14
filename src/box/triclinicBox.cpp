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

#include <memory>
#include <mstd/debugging.hpp>
#include <mstd/math.hpp>

#include "box.hpp"
#include "defaults.hpp"
#include "staticMatrix.hpp"
#include "unit_types.hpp"
#include "units.hpp"
#include "vector3d.hpp"

namespace molsys
{

    /**
     * @brief Construct a new Triclinic Box object
     *
     * @param boxDimensions
     * @param boxAngles
     * @param allowChangeOfAngles Whether the box is allowed to change its
     * angles during simulation.
     */
    TriclinicBox::TriclinicBox(
        const Pos3D         &boxDimensions,
        const linalg::Vec3D &boxAngles,
        bool                 allowChangeOfAngles
    )
        : Box(boxDimensions, boxAngles),
          _allowChangeOfAngles(allowChangeOfAngles)
    {
    }

    /** @brief Copy the cell geometry into an independent box. */
    std::unique_ptr<Box> TriclinicBox::clone() const
    {
        return std::make_unique<TriclinicBox>(*this);
    }

    /**
     * @brief Calculate the volume of the box
     *
     * @return volume
     */
    Volume TriclinicBox::_calculateVolume(const PosTensor3D &boxMatrix)
    {
        return static_cast<Volume>(det(boxMatrix));
    }

    /**
     * @brief Get the volume of the box (cached)
     *
     * @return volume
     */
    Volume TriclinicBox::getVolume() const { return _volume.get(*this); }

    /**
     * @brief Calculate the box matrix from the box dimensions and angles
     *
     */
    PosTensor3D TriclinicBox::_calculateBoxMatrix(
        const Pos3D            &boxDimensions,
        const QuantityTensor3D &transformationMatrix
    )
    {
        PosTensor3D boxMatrix;

        boxMatrix[0] = boxDimensions * transformationMatrix[0];
        boxMatrix[1] = boxDimensions * transformationMatrix[1];
        boxMatrix[2] = boxDimensions * transformationMatrix[2];

        return boxMatrix;
    }

    /**
     * @brief Calculate the rotation matrix
     *
     * @param boxAngles
     */
    QuantityTensor3D TriclinicBox::_calculateTransformationMatrix(
        const Angles3D &boxAngles
    )
    {
        const auto cosGamma = _cosGamma(boxAngles);
        const auto sinGamma = _sinGamma(boxAngles);
        const auto cosBeta  = _cosBeta(boxAngles);

        QuantityTensor3D transformationMatrix{0.0 * one};

        transformationMatrix[0][0] = 1;
        transformationMatrix[0][1] = cosGamma;
        transformationMatrix[0][2] = cosBeta;

        transformationMatrix[1][1] = sinGamma;
        transformationMatrix[1][2] =
            _cosAlpha(boxAngles) - (cosBeta * cosGamma);
        transformationMatrix[1][2] /= sinGamma;

        const auto cosAngles = linalg::cos(boxAngles);
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
    InvPosTensor3D TriclinicBox::_calculateInverse(const PosTensor3D &boxMatrix)
    {
        return inverse(boxMatrix);
    }

    /**
     * @brief Calculate the inverse of a dimensionless (transformation) matrix
     *
     */
    QuantityTensor3D TriclinicBox::_calculateInverse(
        const QuantityTensor3D &matrix
    )
    {
        return inverse(matrix);
    }

    /**
     * @brief applies the periodic boundary conditions
     *
     * @param position
     */
    void TriclinicBox::applyPBC(Pos3D &position) const
    {
        const auto originalPosition = position;

        auto fractionalPosition = _inverse.get(*this) * position;

        fractionalPosition -= round(fractionalPosition);

        const auto boxMatrix = _boxMatrix.get(*this);
        position             = boxMatrix * fractionalPosition;

        const auto distance = norm(position);

        Pos3D analyticPosition   = position;
        Pos   analyticalDistance = distance;

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
                            boxMatrix * Quantity3D{
                                            static_cast<double>(i) * one,
                                            static_cast<double>(j) * one,
                                            static_cast<double>(k) * one,
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
     * @return Pos3D wrapped position with box centered at origin
     */
    Pos3D TriclinicBox::wrapPositionIntoBox(const Pos3D &pos) const
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
     * @return Pos3D
     */
    Pos3D TriclinicBox::calcShiftVector(const Pos3D &vec) const
    {
        return _boxMatrix.get(*this) * round(_inverse.get(*this) * vec);
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
    void TriclinicBox::scaleBox(const QuantityTensor3D &scalingTensor)
    {
        if (!_allowChangeOfAngles)
        {
            _set_boxDimensions(
                diagonal(scalingTensor).numerical_value_in(one) *
                get_boxDimensions()
            );
        }
        else
        {
            const auto boxMatrix =
                scalingTensor.numerical_value_in(one) * _boxMatrix.get(*this);

            const auto &[boxDimensions, boxAngles] =
                calcBoxDimAndAnglesFromBoxMatrix(boxMatrix);

            _set_boxDimensions(boxDimensions);
            _set_angles(boxAngles);
        }
    }

    /**
     * @brief determine box dimensions and angles from box matrix (angles are in
     * RAD)
     *
     * @param boxMatrix
     * @return std::pair<Pos3D, linalg::Vec3D>
     */
    std::pair<Pos3D, Angles3D> calcBoxDimAndAnglesFromBoxMatrix(
        const PosTensor3D &boxMatrix
    )
    {
        const auto box_x = boxMatrix[0][0];
        const auto box_y = mstd::sqrt(
            (boxMatrix[1][1] * boxMatrix[1][1]) +
            (boxMatrix[0][1] * boxMatrix[0][1])
        );
        const auto box_z = mstd::sqrt(
            (boxMatrix[2][2] * boxMatrix[2][2]) +
            (boxMatrix[1][2] * boxMatrix[1][2]) +
            (boxMatrix[0][2] * boxMatrix[0][2])
        );

        const auto cos_alpha = (boxMatrix[0][1] * boxMatrix[0][2] +
                                boxMatrix[1][1] * boxMatrix[1][2]) /
                               (box_y * box_z);

        const auto cos_beta  = boxMatrix[0][2] / box_z;
        const auto cos_gamma = boxMatrix[0][1] / box_y;

        const auto alpha = static_cast<Angle>(mp_units::si::acos(cos_alpha));
        const auto beta  = static_cast<Angle>(mp_units::si::acos(cos_beta));
        const auto gamma = static_cast<Angle>(mp_units::si::acos(cos_gamma));

        return {Pos3D{box_x, box_y, box_z}, {alpha, beta, gamma}};
    }

    /**
     * @brief get the minimal box dimension
     *
     * @return Pos
     */
    Pos TriclinicBox::getMinimalBoxDimension() const
    {
        return min(diagonal(_boxMatrix.get(*this)));
    }

    /**
     * @brief calculate cos of alpha
     *
     * @return double
     */
    double TriclinicBox::_cosAlpha(const Angles3D &boxAngles)
    {
        return mstd::cos(boxAngles[0]);
    }

    /**
     * @brief calculate cos of beta
     *
     * @return double
     */
    double TriclinicBox::_cosBeta(const Angles3D &boxAngles)
    {
        return mstd::cos(boxAngles[1]);
    }

    /**
     * @brief calculate cos of gamma
     *
     * @return double
     */
    double TriclinicBox::_cosGamma(const Angles3D &boxAngles)
    {
        return mstd::cos(boxAngles[2]);
    }

    /**
     * @brief calculate sin of alpha
     *
     * @return double
     */
    double TriclinicBox::_sinAlpha(const Angles3D &boxAngles)
    {
        return mstd::sin(boxAngles[0]);
    }

    /**
     * @brief calculate sin of beta
     *
     * @return double
     */
    double TriclinicBox::_sinBeta(const Angles3D &boxAngles)
    {
        return mstd::sin(boxAngles[1]);
    }

    /**
     * @brief calculate sin of gamma
     *
     * @return double
     */
    double TriclinicBox::_sinGamma(const Angles3D &boxAngles)
    {
        return mstd::sin(boxAngles[2]);
    }

    /**
     * @brief get the box matrix
     *
     * @return PosTensor3D
     */
    PosTensor3D TriclinicBox::getBoxMatrix(Periodicity per) const
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
                boxMatrix[0][1] = Pos::zero();        // Clear XY cross term
                boxMatrix[0][2] = Pos::zero();        // Clear XZ cross term
                boxMatrix[1][0] = Pos::zero();        // Clear YX cross term
                boxMatrix[1][2] = Pos::zero();        // Clear YZ cross term
                boxMatrix[2][0] = Pos::zero();        // Clear ZX cross term
                boxMatrix[2][1] = Pos::zero();        // Clear ZY cross term
                break;
            case Periodicity::X:
                boxMatrix[1][1] =
                    defaults::VACUUM_BOX_DIMENSION;   // Y dimension
                boxMatrix[2][2] =
                    defaults::VACUUM_BOX_DIMENSION;   // Z dimension
                boxMatrix[0][1] = Pos::zero();        // Clear XY cross term
                boxMatrix[0][2] = Pos::zero();        // Clear XZ cross term
                boxMatrix[1][0] = Pos::zero();        // Clear YX cross term
                boxMatrix[1][2] = Pos::zero();        // Clear YZ cross term
                boxMatrix[2][0] = Pos::zero();        // Clear ZX cross term
                boxMatrix[2][1] = Pos::zero();        // Clear ZY cross term
                break;
            case Periodicity::Y:
                boxMatrix[0][0] =
                    defaults::VACUUM_BOX_DIMENSION;   // X dimension
                boxMatrix[2][2] =
                    defaults::VACUUM_BOX_DIMENSION;   // Z dimension
                boxMatrix[0][1] = Pos::zero();        // Clear XY cross term
                boxMatrix[0][2] = Pos::zero();        // Clear XZ cross term
                boxMatrix[1][0] = Pos::zero();        // Clear YX cross term
                boxMatrix[1][2] = Pos::zero();        // Clear YZ cross term
                boxMatrix[2][0] = Pos::zero();        // Clear ZX cross term
                boxMatrix[2][1] = Pos::zero();        // Clear ZY cross term
                break;
            case Periodicity::Z:
                boxMatrix[0][0] =
                    defaults::VACUUM_BOX_DIMENSION;   // X dimension
                boxMatrix[1][1] =
                    defaults::VACUUM_BOX_DIMENSION;   // Y dimension
                boxMatrix[0][1] = Pos::zero();        // Clear XY cross term
                boxMatrix[0][2] = Pos::zero();        // Clear XZ cross term
                boxMatrix[1][0] = Pos::zero();        // Clear YX cross term
                boxMatrix[1][2] = Pos::zero();        // Clear YZ cross term
                boxMatrix[2][0] = Pos::zero();        // Clear ZX cross term
                boxMatrix[2][1] = Pos::zero();        // Clear ZY cross term
                break;
            case Periodicity::XY:
                boxMatrix[2][2] =
                    defaults::VACUUM_BOX_DIMENSION;   // Z dimension
                boxMatrix[0][2] = Pos::zero();        // Clear XZ cross term
                boxMatrix[1][2] = Pos::zero();        // Clear YZ cross term
                boxMatrix[2][0] = Pos::zero();        // Clear ZX cross term
                boxMatrix[2][1] = Pos::zero();        // Clear ZY cross term
                break;
            case Periodicity::XZ:
                boxMatrix[1][1] =
                    defaults::VACUUM_BOX_DIMENSION;   // Y dimension
                boxMatrix[0][1] = Pos::zero();        // Clear XY cross term
                boxMatrix[1][0] = Pos::zero();        // Clear YX cross term
                boxMatrix[1][2] = Pos::zero();        // Clear YZ cross term
                boxMatrix[2][1] = Pos::zero();        // Clear ZY cross term
                break;
            case Periodicity::YZ:
                boxMatrix[0][0] =
                    defaults::VACUUM_BOX_DIMENSION;   // X dimension
                boxMatrix[0][1] = Pos::zero();        // Clear XY cross term
                boxMatrix[0][2] = Pos::zero();        // Clear XZ cross term
                boxMatrix[1][0] = Pos::zero();        // Clear YX cross term
                boxMatrix[2][0] = Pos::zero();        // Clear ZX cross term
                break;
            case Periodicity::XYZ: break;
        }

        return boxMatrix;
    }

    /**
     * @brief get the box matrix without specifying periodicity
     *
     * @return PosTensor3D
     */
    PosTensor3D TriclinicBox::getBoxMatrix() const
    {
        return _boxMatrix.get(*this);
    }

    /**
     * @brief get the inverse of the box matrix
     *
     * @return InvPosTensor3D
     */
    InvPosTensor3D TriclinicBox::getInverseBoxMatrix() const
    {
        return _inverse.get(*this);
    }

    /**
     * @brief indicate whether the box allows change of angles
     *
     * @return true if change of angles is allowed, false otherwise
     */
    bool TriclinicBox::allowChangeOfAngles() const
    {
        return _allowChangeOfAngles;
    }

    /**
     * @brief indicate whether the box is orthorhombic
     *
     * @return true if the box is orthorhombic, false otherwise
     */
    bool TriclinicBox::isOrthorhombic() const { return false; }
}   // namespace molsys
