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

/**
 * @file box.hpp
 * @author Jakob Gamper (97gamjak@gmail.com)
 * @brief This file contains the main class definition for the box. The class is
 * used to represent the base class for the simulation cell. The class is used
 * to calculate the volume of the simulation cell, apply periodic boundary
 * conditions and scale the simulation cell. The two derived classes are the
 * orthorhombic and triclinic box.
 *
 * @date 2024-12-09
 *
 */

#include "staticMatrix.hpp"   // for StaticMatrix3x3
#include "typeAliases.hpp"    // for diagonalMatrix
#include "vector3d.hpp"       // for Vec3D

namespace simulationBox
{
    /**
     * @class Box
     *
     * @brief This class represents the unit cell of a general triclinic box
     *
     */
    class Box
    {
       protected:
        pq::Vec3D _boxDimensions;

        bool   _boxSizeHasChanged = false;
        double _volume;

#ifndef __PQ_LEGACY__
        std::vector<Real> _boxParams;

#ifdef __PQ_GPU__
        Real *_boxParamsDevice;
#endif

#endif
       public:
        virtual ~Box() = default;   // change this to free the box params

        virtual void      applyPBC(pq::Vec3D &position) const          = 0;
        virtual void      scaleBox(const pq::tensor3D &scalingFactors) = 0;
        virtual double    calculateVolume()                            = 0;
        virtual pq::Vec3D calcShiftVector(const pq::Vec3D &) const     = 0;

        virtual bool isOrthoRhombic() const = 0;

#ifndef __PQ_LEGACY__
        virtual void updateBoxParams() = 0;
#endif

#ifdef __PQ_GPU__
        void initDeviceMemory(device::Device &device);
        void copyBoxParamsTo(pq::Device &device);
#endif

        /*****************************************************
         * virtual methods that are overriden in triclinicBox *
         ******************************************************/

        virtual void setBoxDimensions(const pq::Vec3D &boxDimensions);

        [[nodiscard]] virtual double getMinimalBoxDimension() const;

        [[nodiscard]] virtual pq::Vec3D    getBoxAngles() const;
        [[nodiscard]] virtual pq::tensor3D getBoxMatrix() const;

        [[nodiscard]] virtual pq::Vec3D toOrthoSpace(const pq::Vec3D &) const;
        [[nodiscard]] virtual pq::tensor3D toOrthoSpace(const pq::tensor3D &)
            const;

        [[nodiscard]] virtual pq::Vec3D    toSimSpace(const pq::Vec3D &) const;
        [[nodiscard]] virtual pq::tensor3D toSimSpace(const pq::tensor3D &)
            const;

        /********************
         * standard getters *
         ********************/

        [[nodiscard]] bool      getBoxSizeHasChanged() const;
        [[nodiscard]] double    getVolume() const;
        [[nodiscard]] pq::Vec3D getBoxDimensions() const;

#ifndef __PQ_LEGACY__
        [[nodiscard]] Real *getBoxParamsPtr();
#endif

        /********************
         * standard setters *
         ********************/

        void setVolume(const double volume);
        void setBoxSizeHasChanged(const bool boxSizeHasChanged);
    };

}   // namespace simulationBox

#endif   // _BOX_HPP_