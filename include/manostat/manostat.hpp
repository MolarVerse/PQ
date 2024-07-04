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

#ifndef _MANOSTAT_HPP_

#define _MANOSTAT_HPP_

#include "manostatSettings.hpp"
#include "staticMatrix3x3.hpp"   // for tensor3D
#include "timer.hpp"             // for Timer

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
    class Manostat : public timings::Timer
    {
       protected:
        linearAlgebra::tensor3D _pressureTensor = {0.0};
        double                  _pressure;
        double _targetPressure;   // no default value, must be set

       public:
        Manostat() = default;
        explicit Manostat(const double targetPressure)
            : _targetPressure(targetPressure)
        {
        }
        virtual ~Manostat() = default;

        void calculatePressure(const simulationBox::SimulationBox &, physicalData::PhysicalData &);
        virtual void applyManostat(simulationBox::SimulationBox &, physicalData::PhysicalData &);

        virtual settings::ManostatType getManostatType() const;
        virtual settings::Isotropy     getIsotropy() const;
    };

}   // namespace manostat

#endif   // _MANOSTAT_HPP_