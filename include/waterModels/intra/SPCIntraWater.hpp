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

using namespace constants;

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
    class SPCIntraWater : public IntraWater
    {
       public:
        void calculate(
            simulationBox::SimulationBox &,
            physicalData::PhysicalData &
        ) final;

        // clang-format off
        virtual double getEqOHDistance() const = 0;          // Angström
        virtual double getEqHOHAngle() const = 0;            // radians
        virtual double getForceConstantOHBond() const = 0;   // kcal mol^-1 Angström^-2
        virtual double getForceConstantHOHAngle() const = 0; // kcal mol^-1 rad^-2
        // clang-format on
    };

    class SPCFwIntraWater : public SPCIntraWater
    {
        // clang-format off
       private:
        static constexpr double _eqOHDistance          = 1.012;                 // Angström
        static constexpr double _eqHOHAngle            = 113.24 * DEG_TO_RAD; // radians
        static constexpr double _forceConstantOHBond   = 1059.162;              // kcal mol^-1 Angström^-2
        static constexpr double _forceConstantHOHAngle = 75.9;                  // kcal mol^-1 rad^-2

       public:
        double getEqOHDistance() const final          { return _eqOHDistance; }         
        double getEqHOHAngle() const final            { return _eqHOHAngle; }           
        double getForceConstantOHBond() const final   { return _forceConstantOHBond; }  
        double getForceConstantHOHAngle() const final { return _forceConstantHOHAngle; }
        // clang-format on
    };

    class qSPCFwIntraWater : public SPCIntraWater
    {
        // clang-format off
       private:
        static constexpr double _eqOHDistance          = 1.0;                  // Angström
        static constexpr double _eqHOHAngle            = 112.0 * DEG_TO_RAD; // radians
        static constexpr double _forceConstantOHBond   = 1059.162;             // kcal mol^-1 Angström^-2
        static constexpr double _forceConstantHOHAngle = 75.9;                 // kcal mol^-1 rad^-2

       public:
        double getEqOHDistance() const final          { return _eqOHDistance; }         
        double getEqHOHAngle() const final            { return _eqHOHAngle; }           
        double getForceConstantOHBond() const final   { return _forceConstantOHBond; }  
        double getForceConstantHOHAngle() const final { return _forceConstantHOHAngle; }
        // clang-format on
    };

}   // namespace waterModel

#ifndef _SPC_INTRA_WATER_TPP_
#include "SPCIntraWater.tpp"   // IWYU pragma: export - DO NOT MOVE THIS LINE
#endif

#endif   //  _SPC_INTRA_WATER_HPP_
