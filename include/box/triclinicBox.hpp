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

#ifndef _TRICLINIC_BOX_HPP_

#define _TRICLINIC_BOX_HPP_

/**
 * @file triclinicBox.hpp
 * @author Jakob Gamper (97gamjak@gmail.com)
 * @brief This file contains the main class definition for the triclinic box.
 * The class is used to represent the unit cell of a triclinic box. The class is
 * used to calculate the volume of the simulation cell, apply periodic boundary
 * conditions and scale the simulation cell. This class is derived from the
 * base class Box.
 *
 * @date 2024-12-09
 *
 * @see box.hpp
 */

#include "box.hpp"           // for Box
#include "typeAliases.hpp"   // for tensor3D, Vec3D

namespace simulationBox
{
    pq::Vec3DPair calcBoxDimAndAnglesFromBoxMatrix(const pq::tensor3D &);

    /**
     * @class TriclinicBox
     *
     * @brief This class represents the unit cell of a triclinic box
     *
     * @see Box
     */
    class TriclinicBox : public Box
    {
       private:
        pq::Vec3D    _boxAngles;
        pq::tensor3D _boxMatrix{0.0};
        pq::tensor3D _transformationMatrix{0.0};

        void calculateBoxMatrix();
        void calculateTransformationMatrix();

       public:
        [[nodiscard]] double calculateVolume() override;

        [[nodiscard]] bool isOrthoRhombic() const override;

        [[nodiscard]] pq::Vec3D calcShiftVector(const pq::Vec3D &)
            const override;

        [[nodiscard]] pq::Vec3D toOrthoSpace(const pq::Vec3D &) const override;
        [[nodiscard]] pq::tensor3D toOrthoSpace(const pq::tensor3D &)
            const override;

        [[nodiscard]] pq::Vec3D    toSimSpace(const pq::Vec3D &) const override;
        [[nodiscard]] pq::tensor3D toSimSpace(const pq::tensor3D &)
            const override;

        void applyPBC(pq::Vec3D &position) const override;
        void scaleBox(const pq::tensor3D &scalingTensor) override;

        void setBoxAngles(const pq::Vec3D &boxAngles);
        void setBoxDimensions(const pq::Vec3D &boxDimensions) override;

        [[nodiscard]] double getMinimalBoxDimension() const override;

        [[nodiscard]] double cosAlpha() const;
        [[nodiscard]] double cosBeta() const;
        [[nodiscard]] double cosGamma() const;
        [[nodiscard]] double sinAlpha() const;
        [[nodiscard]] double sinBeta() const;
        [[nodiscard]] double sinGamma() const;

        [[nodiscard]] pq::Vec3D    getBoxAngles() const override;
        [[nodiscard]] pq::tensor3D getBoxMatrix() const override;
        [[nodiscard]] pq::tensor3D getTransformationMatrix() const;

        void flattenBoxParams() override;
        void deFlattenBoxParams() override;
    };

}   // namespace simulationBox

#ifndef __TRICLINIC_BOX_INL__
    #include "triclinicBox.inl"   // IWYU pragma: keep
#endif

#endif   // _TRICLINIC_BOX_HPP_