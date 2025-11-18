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

#ifndef _SPC_INTRA_WATER_HPP_

#define _SPC_INTRA_WATER_HPP_

#include "intraWater.hpp"
#include "physicalData.hpp"    // for PhysicalData
#include "simulationBox.hpp"   // for SimulationBox
#include "typeAliases.hpp"

namespace waterModel
{
    template <class Derived>
    class SPCIntraWater : public IntraWater
    {
       public:
        virtual void calculate(pq::SimBox&, pq::PhysicalData&) override;
    };

    class SPCFwIntraParam : public SPCIntraWater<SPCFwIntraParam>
    {
       private:
        static constexpr auto _eqOHDistance          = 1.012;
        static constexpr auto _eqHOHAngle            = 113.24;
        static constexpr auto _forceConstantOHBond   = 1059.162;
        static constexpr auto _forceConstantHOHAngle = 75.9;

        friend class SPCIntraWater<SPCFwIntraParam>;
    };

    class qSPCFwIntraParam : public SPCIntraWater<qSPCFwIntraParam>
    {
       private:
        static constexpr auto _eqOHDistance          = 1.0;
        static constexpr auto _eqHOHAngle            = 112.0;
        static constexpr auto _forceConstantOHBond   = 1059.162;
        static constexpr auto _forceConstantHOHAngle = 75.9;

        friend class SPCIntraWater<qSPCFwIntraParam>;
    };

}   // namespace waterModel

#include "SPCIntraWater.tpp.hpp"   // DO NOT MOVE THIS LINE

#endif   //  _SPC_INTRA_WATER_HPP_