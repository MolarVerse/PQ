/*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
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

#include "staticMatrix3x3.hpp"   // for StaticMatrix3x3
#include "vector3d.hpp"          // for Vec3D

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
        linearAlgebra::Vec3D _boxDimensions;

        bool   _boxSizeHasChanged = false;
        double _volume;

      public:
        virtual ~Box() = default;

        void scaleBox(const linearAlgebra::Vec3D &scalingFactors);

        [[nodiscard]] virtual double               calculateVolume()                                                = 0;
        [[nodiscard]] virtual linearAlgebra::Vec3D calculateShiftVector(const linearAlgebra::Vec3D &position) const = 0;
        virtual void                               applyPBC(linearAlgebra::Vec3D &position) const                   = 0;

        /*****************************************************
         * virtual methods that are overriden in triclinicBox *
         ******************************************************/

        [[nodiscard]] virtual linearAlgebra::Vec3D                   getBoxAngles() const { return linearAlgebra::Vec3D(90.0); }
        [[nodiscard]] virtual linearAlgebra::StaticMatrix3x3<double> getBoxMatrix() const
        {
            return diagonalMatrix(_boxDimensions);
        }

        [[nodiscard]] virtual linearAlgebra::Vec3D transformIntoOrthogonalSpace(const linearAlgebra::Vec3D &position) const
        {
            return position;
        }
        [[nodiscard]] virtual linearAlgebra::Vec3D transformIntoSimulationSpace(const linearAlgebra::Vec3D &position) const
        {
            return position;
        }

        virtual void setBoxDimensions(const linearAlgebra::Vec3D &boxDimensions) { _boxDimensions = boxDimensions; }

        /********************
         * standard getters *
         ********************/

        [[nodiscard]] bool getBoxSizeHasChanged() const { return _boxSizeHasChanged; }

        [[nodiscard]] double getVolume() const { return _volume; }
        [[nodiscard]] double getMinimalBoxDimension() const { return minimum(_boxDimensions); }

        [[nodiscard]] linearAlgebra::Vec3D getBoxDimensions() const { return _boxDimensions; }

        /********************
         * standard setters *
         ********************/

        void setVolume(const double volume) { _volume = volume; }
        void setBoxSizeHasChanged(const bool boxSizeHasChanged) { _boxSizeHasChanged = boxSizeHasChanged; }
    };

}   // namespace simulationBox

#endif   // _BOX_HPP_