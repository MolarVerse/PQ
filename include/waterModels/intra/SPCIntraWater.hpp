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

#include "constants/conversionFactors.hpp"   // for constants
#include "intraWater.hpp"                    // for IntraWater
#include "physicalData.hpp"                  // for PhysicalData
#include "simulationBox.hpp"                 // for SimulationBox
#include "typeAliases.hpp"

using namespace constants;

namespace waterModel
{
    class SPCIntraWaterParam
    {
       public:
        double eqOHDistance{};            // Angström
        double eqHOHAngle{};              // radians
        double forceConstantOHBond{};     // kcal mol^-1 Angström^-2
        double forceConstantHOHAngle{};   // kcal mol^-1 rad^-2

        SPCIntraWaterParam() = delete;

        constexpr SPCIntraWaterParam(
            const double eqOHDist,
            const double eqHOHAng,
            const double forceConstOHBond,
            const double forceConstHOHAngle
        ) noexcept
            : eqOHDistance{eqOHDist},
              eqHOHAngle{eqHOHAng},
              forceConstantOHBond{forceConstOHBond},
              forceConstantHOHAngle{forceConstHOHAngle}
        {
        }
    };

    class SPCIntraWater : public IntraWater
    {
       public:
        void calculate(pq::SimBox &, pq::PhysicalData &) final;

       private:
        virtual const SPCIntraWaterParam &get_parameters() const = 0;
    };

    class SPCFwIntraWater : public SPCIntraWater
    {
       private:
        // clang-format off
        const SPCIntraWaterParam _parameters{
            1.012,                 // eqOHDistance in Angström
            113.24 * _DEG_TO_RAD_, // eqHOHAngle in radians
            1059.162,              // forceConstantOHBond in kcal mol^-1 Angström^-2
            75.9                   // forceConstantHOHAngle in kcal mol^-1 rad^-2
        };
        // clang-format on

       public:
        const SPCIntraWaterParam &get_parameters() const final
        {
            return _parameters;
        }
    };

    class qSPCFwIntraWater : public SPCIntraWater
    {
       private:
        // clang-format off
        const SPCIntraWaterParam _parameters{
            1.0,                  // eqOHDistance in Angström
            112.0 * _DEG_TO_RAD_, // eqHOHAngle in radians
            1059.162,             // forceConstantOHBond in kcal mol^-1 Angström^-2
            75.9                  // forceConstantHOHAngle in kcal mol^-1 rad^-2
        };
        // clang-format on

       public:
        const SPCIntraWaterParam &get_parameters() const final
        {
            return _parameters;
        }
    };

}   // namespace waterModel

#include "SPCIntraWater.tpp.hpp"   // DO NOT MOVE THIS LINE

#endif   //  _SPC_INTRA_WATER_HPP_