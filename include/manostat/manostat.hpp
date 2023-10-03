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

#ifndef _MANOSTAT_HPP_

#define _MANOSTAT_HPP_

#include "vector3d.hpp"   // for Vec3D

namespace physicalData
{
    class PhysicalData;   // forward declaration
}

namespace simulationBox
{
    class SimulationBox;   // forward declaration
}

namespace manostat
{
    /**
     * @class Manostat
     *
     * @brief Manostat is a base class for all manostats
     *
     */
    class Manostat
    {
      protected:
        linearAlgebra::Vec3D _pressureVector = {0.0, 0.0, 0.0};
        double               _pressure;
        double               _targetPressure;   // no default value, must be set

      public:
        Manostat() = default;
        explicit Manostat(const double targetPressure) : _targetPressure(targetPressure) {}
        virtual ~Manostat() = default;

        void         calculatePressure(const simulationBox::SimulationBox &, physicalData::PhysicalData &);
        virtual void applyManostat(simulationBox::SimulationBox &, physicalData::PhysicalData &);
    };

}   // namespace manostat

#endif   // _MANOSTAT_HPP_