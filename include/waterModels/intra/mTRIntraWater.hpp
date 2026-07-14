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
    class MTRIntraWaterParam
    {
       public:
        virtual double getEqOHDistance() const noexcept = 0;   // Angström
        virtual double getEqHHDistance() const noexcept = 0;   // Angström
        virtual double getDOH() const noexcept          = 0;   // kcal mol^-1
        virtual double getAlpha() const noexcept        = 0;   // Angström^-1
        virtual double getBeta() const noexcept         = 0;   // Angström^-2
        virtual double getLtt() const noexcept = 0;   // kcal mol^-1 Angström^-2
        virtual double getLrt() const noexcept = 0;   // kcal mol^-1 Angström^-2
        virtual double getLrr() const noexcept = 0;   // kcal mol^-1 Angström^-2
    };

    class MTRIntraWater : public IntraWater
    {
       private:
        std::unique_ptr<MTRIntraWaterParam> _parameter;

       public:
        MTRIntraWater() = delete;
        MTRIntraWater(std::unique_ptr<MTRIntraWaterParam> parameter)
            : _parameter{std::move(parameter)}
        {
        }

        void calculate(pq::SimBox &, pq::PhysicalData &) final;
    };

    class SPCMTRIntraWaterParam : public MTRIntraWaterParam
    {
       public:
        // clang-format off
        double getEqOHDistance() const noexcept final { return 1.0; }         // Angström
        double getEqHHDistance() const noexcept final { return 1.632993162; } // Angström
        double getDOH() const noexcept final { return 101.9048757170172; }    // kcal mol^-1
        double getAlpha() const noexcept final { return 2.511; }              // Angström^-1
        double getBeta() const noexcept final { return 3.0; }                 // Angström^-2
        double getLtt() const noexcept final { return 264.5841300191204; }    // kcal mol^-1 Angström^-2
        double getLrt() const noexcept final { return -211.0444550669216; }   // kcal mol^-1 Angström^-2
        double getLrr() const noexcept final { return 155.7839388145315; }    // kcal mol^-1 Angström^-2
        // clang-format on
    };

    class TIP3PMTRIntraWaterParam : public MTRIntraWaterParam
    {
       public:
        // clang-format off
        double getEqOHDistance() const noexcept final { return  0.9572; }    // Angström
        double getEqHHDistance() const noexcept final { return 1.5139; }     // Angström
        double getDOH() const noexcept final { return   101.9048757170172; } // kcal mol^-1
        double getAlpha() const noexcept final { return 2.483; }             // Angström^-1
        double getBeta() const noexcept final { return 3.0; }                // Angström^-2
        double getLtt() const noexcept final { return 235.2449808795411; }   // kcal mol^-1 Angström^-2
        double getLrt() const noexcept final { return -181.2906309751434; }  // kcal mol^-1 Angström^-2
        double getLrr() const noexcept final { return 127.1534416826004; }   // kcal mol^-1 Angström^-2
        // clang-format on
    };

}   // namespace waterModel

#include "mTRIntraWater.tpp.hpp"   // DO NOT MOVE THIS LINE

#endif   //  _MTR_INTRA_WATER_HPP_