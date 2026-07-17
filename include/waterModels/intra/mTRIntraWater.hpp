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
    class MTRIntraWater : public IntraWater
    {
       public:
        void calculate(pq::SimBox &, pq::PhysicalData &) final;

        virtual double getEqOHDistance() const noexcept = 0;   // Angström
        virtual double getEqHHDistance() const noexcept = 0;   // Angström
        virtual double getDOH() const noexcept          = 0;   // kcal mol^-1
        virtual double getAlpha() const noexcept        = 0;   // Angström^-1
        virtual double getBeta() const noexcept         = 0;   // Angström^-2
        virtual double getLtt() const noexcept = 0;   // kcal mol^-1 Angström^-2
        virtual double getLrt() const noexcept = 0;   // kcal mol^-1 Angström^-2
        virtual double getLrr() const noexcept = 0;   // kcal mol^-1 Angström^-2
    };

    class SPCMTRIntraWater : public MTRIntraWater
    {
        // clang-format off
        private:
        static constexpr double eqOHDistance = 1.0;         // Angström
        static constexpr double eqHHDistance = 1.632993162; // Angström
        static constexpr double dOH = 101.9048757170172;    // kcal mol^-1
        static constexpr double alpha = 2.511;              // Angström^-1
        static constexpr double beta = 3.0;                 // Angström^-2
        static constexpr double Ltt = 264.5841300191204;    // kcal mol^-1 Angström^-2
        static constexpr double Lrt = -211.0444550669216;   // kcal mol^-1 Angström^-2
        static constexpr double Lrr = 155.7839388145315;    // kcal mol^-1 Angström^-2

       public:
        double getEqOHDistance() const noexcept final { return eqOHDistance; }
        double getEqHHDistance() const noexcept final { return eqHHDistance; }
        double getDOH() const noexcept final { return dOH; }
        double getAlpha() const noexcept final { return alpha; }
        double getBeta() const noexcept final { return beta; }
        double getLtt() const noexcept final { return Ltt; }
        double getLrt() const noexcept final { return Lrt; }
        double getLrr() const noexcept final { return Lrr; }
        // clang-format on
    };

    class TIP3PMTRIntraWater : public MTRIntraWater
    {
        // clang-format off
        private:
        static constexpr double eqOHDistance = 0.9572;    // Angström
        static constexpr double eqHHDistance = 1.5139;    // Angström
        static constexpr double dOH = 101.9048757170172;  // kcal mol^-1
        static constexpr double alpha = 2.483;            // Angström^-1
        static constexpr double beta = 3.0;               // Angström^-2
        static constexpr double Ltt = 235.2449808795411;  // kcal mol^-1 Angström^-2
        static constexpr double Lrt = -181.2906309751434; // kcal mol^-1 Angström^-2
        static constexpr double Lrr = 127.1534416826004;  // kcal mol^-1 Angström^-2

       public:
        double getEqOHDistance() const noexcept final { return eqOHDistance; }
        double getEqHHDistance() const noexcept final { return eqHHDistance; }
        double getDOH() const noexcept final { return dOH; }
        double getAlpha() const noexcept final { return alpha; }
        double getBeta() const noexcept final { return beta; }
        double getLtt() const noexcept final { return Ltt; }
        double getLrt() const noexcept final { return Lrt; }
        double getLrr() const noexcept final { return Lrr; }
        // clang-format on
    };

}   // namespace waterModel

#include "mTRIntraWater.tpp.hpp"   // DO NOT MOVE THIS LINE

#endif   //  _MTR_INTRA_WATER_HPP_