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

#ifndef _TRICLINIC_BOX_HPP_

#define _TRICLINIC_BOX_HPP_

#include "box.hpp"         // for Box
#include "constants.hpp"   // for _DEG_TO_RAD_

namespace simulationBox
{
    /**
     * @class TriclinicBox
     *
     * @brief This class represents the unit cell of a triclinic box
     *
     */
    class TriclinicBox : public Box
    {
      private:
        linearAlgebra::Vec3D                   _boxAngles;
        linearAlgebra::StaticMatrix3x3<double> _boxMatrix;
        linearAlgebra::StaticMatrix3x3<double> _transformationMatrix;

      public:
        [[nodiscard]] double               calculateVolume() override;
        [[nodiscard]] linearAlgebra::Vec3D calculateShiftVector(const linearAlgebra::Vec3D &position) const override;
        [[nodiscard]] linearAlgebra::Vec3D transformIntoOrthogonalSpace(const linearAlgebra::Vec3D &position) const override;
        [[nodiscard]] linearAlgebra::Vec3D transformIntoSimulationSpace(const linearAlgebra::Vec3D &position) const override;

        void applyPBC(linearAlgebra::Vec3D &position) const override;

        void calculateBoxMatrix();
        void calculateTransformationMatrix();

        void setBoxAngles(const linearAlgebra::Vec3D &boxAngles);
        void setBoxDimensions(const linearAlgebra::Vec3D &boxDimensions) override;

        [[nodiscard]] double cosAlpha() const { return ::cos(_boxAngles[0]); }
        [[nodiscard]] double cosBeta() const { return ::cos(_boxAngles[1]); }
        [[nodiscard]] double cosGamma() const { return ::cos(_boxAngles[2]); }
        [[nodiscard]] double sinAlpha() const { return ::sin(_boxAngles[0]); }
        [[nodiscard]] double sinBeta() const { return ::sin(_boxAngles[1]); }
        [[nodiscard]] double sinGamma() const { return ::sin(_boxAngles[2]); }

        [[nodiscard]] linearAlgebra::Vec3D getBoxAngles() const override { return _boxAngles * constants::_DEG_TO_RAD_; }
        [[nodiscard]] linearAlgebra::StaticMatrix3x3<double> getBoxMatrix() const override { return _boxMatrix; }
    };

}   // namespace simulationBox

#endif   // _TRICLINIC_BOX_HPP_