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

#ifndef _BOX_HPP_

#define _BOX_HPP_

#include "staticMatrix.hpp"   // for tensor3D
#include "vector3d.hpp"       // for Vec3D

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

    /**
     * @class Box
     *
     * @brief This class represents the unit cell of a general triclinic box
     *
     */
    class Box
    {
       protected:
        linalg::Vec3D _boxDimensions;

        bool   _boxSizeHasChanged = false;
        double _volume;

       public:
        virtual ~Box() = default;

        virtual void applyPBC(linalg::Vec3D &position) const = 0;

        [[nodiscard]]
        virtual linalg::Vec3D wrapPositionIntoBox(
            const linalg::Vec3D &
        ) const = 0;

        virtual void scaleBox(const linalg::tensor3D &scalingFactors) = 0;

        [[nodiscard]] virtual double calculateVolume() = 0;

        [[nodiscard]]
        virtual linalg::Vec3D calcShiftVector(const linalg::Vec3D &) const = 0;

        /*****************************************************
         * virtual methods that are overriden in triclinicBox *
         ******************************************************/

        virtual void setBoxDimensions(const linalg::Vec3D &boxDimensions);

        [[nodiscard]] virtual double getMinimalBoxDimension() const;

        [[nodiscard]] virtual linalg::Vec3D    getBoxAngles() const;
        [[nodiscard]] virtual linalg::tensor3D getBoxMatrix() const;
        [[nodiscard]]
        virtual linalg::tensor3D getBoxMatrix(Periodicity per) const;

        [[nodiscard]]
        virtual linalg::Vec3D toOrthoSpace(const linalg::Vec3D &) const;
        [[nodiscard]]
        virtual linalg::tensor3D toOrthoSpace(const linalg::tensor3D &) const;

        [[nodiscard]]
        virtual linalg::Vec3D toSimSpace(const linalg::Vec3D &) const;
        [[nodiscard]]
        virtual linalg::tensor3D toSimSpace(const linalg::tensor3D &) const;

        /********************
         * standard getters *
         ********************/

        [[nodiscard]] bool          getBoxSizeHasChanged() const;
        [[nodiscard]] double        getVolume() const;
        [[nodiscard]] linalg::Vec3D getBoxDimensions() const;

        /********************
         * standard setters *
         ********************/

        void setVolume(double volume);
        void setBoxSizeHasChanged(bool boxSizeHasChanged);
    };

}   // namespace molsys

#endif   // _BOX_HPP_
