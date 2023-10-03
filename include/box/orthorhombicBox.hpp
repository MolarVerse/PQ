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

#ifndef _ORTHORHOMBIC_BOX_HPP_

#define _ORTHORHOMBIC_BOX_HPP_

#include "box.hpp"   // for Box

namespace simulationBox
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
        [[nodiscard]] double calculateVolume() override;

        void applyPBC(linearAlgebra::Vec3D &position) const override;

        [[nodiscard]] linearAlgebra::Vec3D calculateShiftVector(const linearAlgebra::Vec3D &position) const override;

        [[nodiscard]] linearAlgebra::Vec3D calculateBoxDimensionsFromDensity(const double totalMass, const double density);
    };

}   // namespace simulationBox

#endif   // _ORTHORHOMBIC_BOX_HPP_