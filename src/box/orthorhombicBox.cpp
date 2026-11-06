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

#include <memory>

#include "constants.hpp"
#include "constants/conversionFactors.hpp"
#include "defaults.hpp"
#include "vector3d.hpp"

namespace molsys
{
    /**
     * @brief Construct a new Orthorhombic Box object
     *
     * @param boxDimensions
     */
    OrthorhombicBox::OrthorhombicBox(const linalg::Vec3D &boxDimensions)
        : Box(boxDimensions, {NINETY_DEG})
    {
    }

    /** @brief Copy the cell geometry into an independent box. */
    std::unique_ptr<Box> OrthorhombicBox::clone() const
    {
        return std::make_unique<OrthorhombicBox>(*this);
    }

    /**
     * @brief Create a vacuum box with default dimensions
     *
     * @return OrthorhombicBox
     */
    OrthorhombicBox OrthorhombicBox::makeVacuumBox()
    {
        return OrthorhombicBox(
            {defaults::VACUUM_BOX_DIMENSION,
             defaults::VACUUM_BOX_DIMENSION,
             defaults::VACUUM_BOX_DIMENSION}
        );
    }

    /**
     * @brief get the volume of the box
     *
     * @return double
     */
    double OrthorhombicBox::getVolume() const
    {
        // NOTE: here we do not use cached values as the overhead is negligible
        // or even negative
        return prod(get_boxDimensions());
    }

    /**
     * @brief applies the periodic boundary conditions
     *
     * @param position
     */
    void OrthorhombicBox::applyPBC(linalg::Vec3D &position) const
    {
        const auto &boxDims  = get_boxDimensions();
        position            -= boxDims * round(position / boxDims);
    }

    /**
     * @brief Calculate the shift vector
     *
     * @param shiftVector
     * @return linalg::Vec3D
     */
    linalg::Vec3D OrthorhombicBox::calcShiftVector(
        const linalg::Vec3D &shiftVector
    ) const
    {
        const auto &boxDims = get_boxDimensions();
        return boxDims * round(shiftVector / boxDims);
    }

    /**
     * @brief Calculate the box dimensions from the density
     *
     * @return vector<double>
     */
    linalg::Vec3D OrthorhombicBox::calcBoxDimFromDensity(
        double totalMass,
        double density
    )
    {
        const auto volume =
            totalMass / (density * KG_PER_L_TO_AMU_PER_ANGSTROM3);

        return {::cbrt(volume)};
    }

    linalg::Vec3D OrthorhombicBox::wrapPositionIntoBox(
        const linalg::Vec3D &pos
    ) const
    {
        linalg::Vec3D position = pos;
        applyPBC(position);
        return position;
    }

    /**
     * @brief scales the cell dimensions and recalculates the volume
     *
     * @param scalingTensor
     */
    void OrthorhombicBox::scaleBox(const linalg::tensor3D &scalingTensor)
    {
        _set_boxDimensions(get_boxDimensions() * diagonal(scalingTensor));
    }

    /**
     * @brief get the minimal box dimension
     *
     * @return double
     */
    double OrthorhombicBox::getMinimalBoxDimension() const
    {
        return minimum(get_boxDimensions());
    }

    /**
     * @brief get the box matrix
     *
     * @return linalg::StaticMatrix3x3<double>
     */
    linalg::StaticMatrix3x3<double> OrthorhombicBox::getBoxMatrix(
        Periodicity periodicity
    ) const
    {
        auto boxMatrix = linalg::diagonalMatrix(get_boxDimensions());

        std::vector<size_t> vacuumIndices{};

        switch (periodicity)
        {
            case Periodicity::NON_PERIODIC: vacuumIndices = {0, 1, 2}; break;
            case Periodicity::X: vacuumIndices = {1, 2}; break;
            case Periodicity::Y: vacuumIndices = {0, 2}; break;
            case Periodicity::Z: vacuumIndices = {0, 1}; break;
            case Periodicity::XY: vacuumIndices = {2}; break;
            case Periodicity::XZ: vacuumIndices = {1}; break;
            case Periodicity::YZ: vacuumIndices = {0}; break;
            case Periodicity::XYZ: break;
        }

        for (auto index : vacuumIndices)
            boxMatrix[index][index] = defaults::VACUUM_BOX_DIMENSION;

        return boxMatrix;
    }

    /**
     * @brief get the box matrix without specifying periodicity
     *
     * @return linalg::StaticMatrix3x3<double>
     */
    linalg::StaticMatrix3x3<double> OrthorhombicBox::getBoxMatrix() const
    {
        return linalg::diagonalMatrix(get_boxDimensions());
    }

    /**
     * @brief get the inverse of the box matrix
     *
     * @return linalg::StaticMatrix3x3<double>
     */
    linalg::StaticMatrix3x3<double> OrthorhombicBox::getInverseBoxMatrix() const
    {
        return linalg::diagonalMatrix(1.0 / get_boxDimensions());
    }

    /**
     * @brief indicate whether the box allows change of angles
     *
     * @return true if change of angles is allowed, false otherwise
     */
    bool OrthorhombicBox::allowChangeOfAngles() const { return false; }

    /**
     * @brief indicate whether the box is orthorhombic
     *
     * @return true if the box is orthorhombic, false otherwise
     */
    bool OrthorhombicBox::isOrthorhombic() const { return true; }
}   // namespace molsys
