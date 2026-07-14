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

#include <memory>
#include <utility>

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
        // clang-format off
        virtual double getEqOHDistance() const noexcept = 0;         // Angström
        virtual double getEqHOHAngle() const noexcept = 0;           // radians
        virtual double getForceConstantOHBond() const noexcept = 0;  // kcal mol^-1 Angström^-2
        virtual double getForceConstantHOHAngle() const noexcept = 0;// kcal mol^-1 rad^-2
        // clang-format on
    };

    class SPCIntraWater : public IntraWater
    {
       private:
        std::unique_ptr<SPCIntraWaterParam> _parameter;

       public:
        SPCIntraWater() = delete;
        SPCIntraWater(std::unique_ptr<SPCIntraWaterParam> parameter)
            : _parameter{std::move(parameter)}
        {
        }

        void calculate(pq::SimBox &, pq::PhysicalData &) final;
    };

    class SPCFwIntraWaterParam : public SPCIntraWaterParam
    {
        // clang-format off
       public:
        double getEqOHDistance() const noexcept final { return 1.012;}               // Angström
        double getEqHOHAngle() const noexcept final { return 113.24 * _DEG_TO_RAD_;} // radians
        double getForceConstantOHBond() const noexcept final { return 1059.162;}     // kcal mol^-1 Angström^-2
        double getForceConstantHOHAngle() const noexcept final { return 75.9;}       // kcal mol^-1 rad^-2
        // clang-format on
    };

    class qSPCFwIntraWaterParam : public SPCIntraWaterParam
    {
        // clang-format off
       public:
        double getEqOHDistance() const noexcept final { return 1.0;}                // Angström
        double getEqHOHAngle() const noexcept final { return 112.0 * _DEG_TO_RAD_;} // radians
        double getForceConstantOHBond() const noexcept final { return 1059.162;}    // kcal mol^-1 Angström^-2
        double getForceConstantHOHAngle() const noexcept final { return 75.9;}      // kcal mol^-1 rad^-2
        // clang-format on
    };

}   // namespace waterModel

#include "SPCIntraWater.tpp.hpp"   // DO NOT MOVE THIS LINE

#endif   //  _SPC_INTRA_WATER_HPP_