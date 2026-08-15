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

#include "intraWater.hpp"   // for IntraWater

namespace waterModel
{
    class MTRIntraWater : public IntraWater
    {
       public:
        void calculate(
            simulationBox::SimulationBox &simBox,
            physicalData::PhysicalData   &physData
        ) final;

        virtual double getEqOHDistance() const = 0;   // Angström
        virtual double getEqHHDistance() const = 0;   // Angström
        virtual double getDOH() const          = 0;   // kcal mol^-1
        virtual double getAlpha() const        = 0;   // Angström^-1
        virtual double getBeta() const         = 0;   // Angström^-2
        virtual double getLtt() const          = 0;   // kcal mol^-1 Angström^-2
        virtual double getLrt() const          = 0;   // kcal mol^-1 Angström^-2
        virtual double getLrr() const          = 0;   // kcal mol^-1 Angström^-2
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
        // clang-format on

       public:
        double getEqOHDistance() const final { return eqOHDistance; }
        double getEqHHDistance() const final { return eqHHDistance; }
        double getDOH() const final { return dOH; }
        double getAlpha() const final { return alpha; }
        double getBeta() const final { return beta; }
        double getLtt() const final { return Ltt; }
        double getLrt() const final { return Lrt; }
        double getLrr() const final { return Lrr; }
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
        // clang-format on

       public:
        double getEqOHDistance() const final { return eqOHDistance; }
        double getEqHHDistance() const final { return eqHHDistance; }
        double getDOH() const final { return dOH; }
        double getAlpha() const final { return alpha; }
        double getBeta() const final { return beta; }
        double getLtt() const final { return Ltt; }
        double getLrt() const final { return Lrt; }
        double getLrr() const final { return Lrr; }
    };

}   // namespace waterModel

#ifndef _MTR_INTRA_WATER_TPP_
#include "mTRIntraWater.tpp"   // IWYU pragma: export - DO NOT MOVE THIS LINE
#endif

#endif   //  _MTR_INTRA_WATER_HPP_
