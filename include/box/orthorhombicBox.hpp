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

#ifndef BOX__ORTHORHOMBIC_BOX_HPP
#define BOX__ORTHORHOMBIC_BOX_HPP

#include <memory>

#include "box.hpp"

namespace molsys
{
    /**
     * @class OrthorhombicBox
     *
     * @brief This class represents the unit cell of an orthorhombic box
     *
     */
    class OrthorhombicBox : public Box
    {
       public:
        explicit OrthorhombicBox(const Pos3D &boxDimensions);

        [[nodiscard]]
        std::unique_ptr<Box> clone() const override;

        [[nodiscard]]
        static OrthorhombicBox makeVacuumBox();

        void applyPBC(Pos3D &position) const override;
        void scaleBox(const QuantityTensor3D &scalingTensor) override;

        [[nodiscard]]
        Pos3D calcShiftVector(const Pos3D &shiftVector) const override;

        [[nodiscard]]
        static Pos3D calcBoxDimFromDensity(
            const Mass    &totalMass,
            const Density &density
        );

        [[nodiscard]]
        Pos3D wrapPositionIntoBox(const Pos3D &pos) const override;

        [[nodiscard]]
        Volume getVolume() const override;

        [[nodiscard]]
        Pos getMinimalBoxDimension() const override;

        [[nodiscard]]
        PosTensor3D getBoxMatrix(Periodicity per) const override;

        [[nodiscard]]
        bool allowChangeOfAngles() const override;

        [[nodiscard]]
        bool isOrthorhombic() const override;

        [[nodiscard]]
        PosTensor3D getBoxMatrix() const override;

        [[nodiscard]]
        InvPosTensor3D getInverseBoxMatrix() const override;
    };

}   // namespace molsys

#endif   // BOX__ORTHORHOMBIC_BOX_HPP
