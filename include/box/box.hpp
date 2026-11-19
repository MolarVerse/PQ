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

#ifndef BOX__BOX_HPP
#define BOX__BOX_HPP

#include <memory>
#include <mstd/types.hpp>
#include <mstd/units.hpp>
#include <mstd/utils.hpp>

#include "unit_types.hpp"
namespace molsys
{
    /**
     * @brief enum Periodicity
     */
    enum class Periodicity : std::uint8_t
    {
        NON_PERIODIC,
        X,
        Y,
        Z,
        XY,
        XZ,
        YZ,
        XYZ
    };

    struct BoxTag
    {
    };
    using BoxVec = mstd::CacheProperty<Pos3D, BoxTag>;

    using AngleVec = mstd::CacheProperty<Angle3D, BoxTag>;

    /**
     * @class Box
     *
     * @brief This class represents the unit cell of a general triclinic box
     *
     */
    class Box
    {
       private:
        BoxVec   _boxDimensions;
        AngleVec _angles;

       public:
        virtual ~Box() = default;

        [[nodiscard]]
        virtual std::unique_ptr<Box> clone() const = 0;

        virtual void applyPBC(Pos3D &position) const = 0;

        [[nodiscard]]
        virtual Pos3D wrapPositionIntoBox(const Pos3D &) const = 0;

        virtual void scaleBox(const QuantityTensor3D &scalingFactors) = 0;

        [[nodiscard]]
        virtual Volume getVolume() const = 0;

        [[nodiscard]]
        virtual Pos3D calcShiftVector(const Pos3D &) const = 0;

        [[nodiscard]]
        virtual Pos getMinimalBoxDimension() const = 0;

        [[nodiscard]]
        virtual PosTensor3D getBoxMatrix(Periodicity per) const = 0;

        [[nodiscard]]
        virtual PosTensor3D getBoxMatrix() const = 0;
        [[nodiscard]]
        virtual InvPosTensor3D getInverseBoxMatrix() const = 0;

        [[nodiscard]]
        virtual bool allowChangeOfAngles() const = 0;

        [[nodiscard]]
        virtual bool isOrthorhombic() const = 0;

        /********************
         * standard getters *
         ********************/

        CONST_PROPERTY_GETTER(_boxDimensions);
        [[nodiscard]] const Angle3D &getBoxAngles() const;

       protected:
        explicit Box(const Pos3D &boxDimensions, const Angle3D &angles);

        static constexpr auto _dimsPtr() { return &Box::_boxDimensions; }
        static constexpr auto _anglesPtr() { return &Box::_angles; }

        INTERNAL_PROPERTY_SETTER(_boxDimensions);
        INTERNAL_PROPERTY_SETTER(_angles);
    };

}   // namespace molsys

#endif   // BOX__BOX_HPP
