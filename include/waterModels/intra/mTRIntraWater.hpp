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

#ifndef _MTR_INTRA_WATER_HPP_

#define _MTR_INTRA_WATER_HPP_

#include "constants/conversionFactors.hpp"   // for constants
#include "intraWater.hpp"                    // for IntraWater
#include "physicalData.hpp"                  // for PhysicalData
#include "simulationBox.hpp"                 // for SimulationBox
#include "typeAliases.hpp"

using namespace constants;

namespace waterModel
{
    class MTRIntraWaterParam
    {
       public:
        double eqOHDistance{};   // Angström
        double eqHHDistance{};   // Angström
        double DOH{};            // kcal mol^-1
        double alpha{};          // Angström^-1
        double beta{};           // Angström^-2
        double Ltt{};            // kcal mol^-1 Angström^-2
        double Lrt{};            // kcal mol^-1 Angström^-2
        double Lrr{};            // kcal mol^-1 Angström^-2

        MTRIntraWaterParam() = delete;

        constexpr MTRIntraWaterParam(
            const double eqOHDist,
            const double eqHHDist,
            const double doh,
            const double alphaAngle,
            const double betaAngle,
            const double ltt,
            const double lrt,
            const double lrr
        ) noexcept
            : eqOHDistance{eqOHDist},
              eqHHDistance{eqHHDist},
              DOH{doh},
              alpha{alphaAngle},
              beta{betaAngle},
              Ltt{ltt},
              Lrt{lrt},
              Lrr{lrr}
        {
        }
    };

    class MTRIntraWater : public IntraWater
    {
       public:
        void calculate(pq::SimBox &, pq::PhysicalData &) final;

       private:
        virtual const MTRIntraWaterParam &get_parameters() const = 0;
    };

    class SPCMTRIntraWater : public MTRIntraWater
    {
       private:
        // clang-format off
        const MTRIntraWaterParam _parameters{
            1.0,                // eqOHDistance in Angström
            1.632993162,        // eqHHDistance in Angström
            101.9048757170172,  // DOH          in kcal mol^-1
            2.511,              // alpha        in Angström^-1
            3.0,                // beta         in Angström^-2
            264.5841300191204,  // Ltt          in kcal mol^-1 Angström^-2
            -211.0444550669216, // Lrt          in kcal mol^-1 Angström^-2
            155.7839388145315   // Lrr          in kcal mol^-1 Angström^-2
        };
        // clang-format on

       public:
        const MTRIntraWaterParam &get_parameters() const final
        {
            return _parameters;
        }
    };

    class TIP3PMTRIntraWater : public MTRIntraWater
    {
       private:
        // clang-format off
        const MTRIntraWaterParam _parameters{
         0.9572,             // eqOHDistance in Angström
         1.5139,             // eqHHDistance in Angström
         101.9048757170172,  // DOH          in kcal mol^-1
         2.483,              // alpha        in Angström^-1
         3.0,                // beta         in Angström^-2
         235.2449808795411,  // Ltt          in kcal mol^-1 Angström^-2
         -181.2906309751434, // Lrt          in kcal mol^-1 Angström^-2
         127.1534416826004   // Lrr          in kcal mol^-1 Angström^-2
        };
        // clang-format on

       public:
        const MTRIntraWaterParam &get_parameters() const final
        {
            return _parameters;
        }
    };

}   // namespace waterModel

#include "mTRIntraWater.tpp.hpp"   // DO NOT MOVE THIS LINE

#endif   //  _MTR_INTRA_WATER_HPP_