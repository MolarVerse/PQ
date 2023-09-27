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

#include "vector3d.hpp"

namespace simulationBox
{
    /**
     * @class Box
     *
     * @brief This class stores all the information about the box.
     *
     * @TODO: think of a way to implement non orthogonal boxes - maybe inheritance?
     *
     */
    class Box
    {
      private:
        linearAlgebra::Vec3D _boxDimensions;
        linearAlgebra::Vec3D _boxAngles = {90.0, 90.0, 90.0};

        double _totalMass;
        double _totalCharge;
        double _density;
        double _volume;

        bool _boxSizeHasChanged = false;

      public:
        double                             calculateVolume();
        [[nodiscard]] linearAlgebra::Vec3D calculateBoxDimensionsFromDensity();

        void applyPBC(linearAlgebra::Vec3D &position) const;
        void scaleBox(const linearAlgebra::Vec3D &scalingFactors);

        /********************
         * standard getters *
         ********************/

        [[nodiscard]] double getMinimalBoxDimension() const { return minimum(_boxDimensions); }

        [[nodiscard]] linearAlgebra::Vec3D getBoxDimensions() const { return _boxDimensions; }
        [[nodiscard]] linearAlgebra::Vec3D getBoxAngles() const { return _boxAngles; }
        [[nodiscard]] double               getTotalMass() const { return _totalMass; }
        [[nodiscard]] double               getTotalCharge() const { return _totalCharge; }
        [[nodiscard]] double               getDensity() const { return _density; }
        [[nodiscard]] double               getVolume() const { return _volume; }
        [[nodiscard]] bool                 getBoxSizeHasChanged() const { return _boxSizeHasChanged; }

        /********************
         * standard setters *
         ********************/

        void setBoxDimensions(const linearAlgebra::Vec3D &boxDimensions) { _boxDimensions = boxDimensions; }
        void setBoxAngles(const linearAlgebra::Vec3D &boxAngles) { _boxAngles = boxAngles; }
        void setDensity(const double density) { _density = density; }
        void setTotalMass(const double totalMass) { _totalMass = totalMass; }
        void setTotalCharge(const double totalCharge) { _totalCharge = totalCharge; }
        void setVolume(const double volume) { _volume = volume; }
        void setBoxSizeHasChanged(const bool boxSizeHasChanged) { _boxSizeHasChanged = boxSizeHasChanged; }
    };

}   // namespace simulationBox

#endif   // _BOX_HPP_