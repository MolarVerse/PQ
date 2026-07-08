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
    struct SPCIntraWaterParam
    {
        double _eqOHDistance          = 0.0;   // Angström
        double _eqHOHAngle            = 0.0;   // radians
        double _forceConstantOHBond   = 0.0;   // kcal mol^-1 Angström^-2
        double _forceConstantHOHAngle = 0.0;   // kcal mol^-1 rad^-2
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
        SPCIntraWaterParam _parameters;

       public:
        SPCFwIntraWater()
        {
            // clang-format off
            _parameters._eqOHDistance          = 1.012;                 // Angström
            _parameters._eqHOHAngle            = 113.24 * _DEG_TO_RAD_; // radians
            _parameters._forceConstantOHBond   = 1059.162;              // kcal mol^-1 Angström^-2
            _parameters._forceConstantHOHAngle = 75.9;                  // kcal mol^-1 rad^-2
            // clang-format on
        }

        const SPCIntraWaterParam &get_parameters() const final
        {
            return _parameters;
        }
    };

    class qSPCFwIntraWater : public SPCIntraWater
    {
       private:
        SPCIntraWaterParam _parameters;

       public:
        qSPCFwIntraWater()
        {
            // clang-format off
            _parameters._eqOHDistance          = 1.0;                 // Angström
            _parameters._eqHOHAngle            = 112.0 * _DEG_TO_RAD_; // radians
            _parameters._forceConstantOHBond   = 1059.162;              // kcal mol^-1 Angström^-2
            _parameters._forceConstantHOHAngle = 75.9;                  // kcal mol^-1 rad^-2
            // clang-format on
        }

        const SPCIntraWaterParam &get_parameters() const final
        {
            return _parameters;
        }
    };

}   // namespace waterModel

#include "SPCIntraWater.tpp.hpp"   // DO NOT MOVE THIS LINE

#endif   //  _SPC_INTRA_WATER_HPP_