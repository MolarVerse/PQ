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

#ifndef _INTRA_WATER_HPP_

#define _INTRA_WATER_HPP_

#include "timer.hpp"

namespace simulationBox
{
    class SimulationBox;   // forward declaration
}   // namespace simulationBox

namespace physicalData
{
    class PhysicalData;   // forward declaration
}   // namespace physicalData

namespace waterModel
{
    class IntraWater : public timings::Timer
    {
       public:
        virtual ~IntraWater() = default;

        virtual void calculate(
            simulationBox::SimulationBox& /*simBox*/,
            physicalData::PhysicalData& /*physData*/
        )
        {
        }
    };

}   // namespace waterModel

#endif   //  _INTRA_WATER_HPP_
