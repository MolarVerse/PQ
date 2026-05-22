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
    template <class Derived>
    class MTRIntraWater : public IntraWater
    {
       public:
        virtual void calculate(pq::SimBox&, pq::PhysicalData&) override final;
    };

    class SPCMTRIntraParam : public MTRIntraWater<SPCMTRIntraParam>
    {
       private:
        // clang-format off
        static constexpr auto _eqOHDistance =    1.0;             // Angström
        static constexpr auto _eqHHDistance =    1.632993162;     // Angström
        static constexpr auto _DOH          =  101.9048757170172; // kcal mol^-1
        static constexpr auto _alpha        =    2.511;           // Angström^-1
        static constexpr auto _beta         =    3.0;             // Angström^-2
        static constexpr auto _Ltt          =  264.5841300191204; // kcal mol^-1 Angström^-2 
        static constexpr auto _Lrt          = -211.0444550669216; // kcal mol^-1 Angström^-2
        static constexpr auto _Lrr          =  155.7839388145315; // kcal mol^-1 Angström^-2
        // clang-format on

        friend class MTRIntraWater<SPCMTRIntraParam>;
    };

    class TIP3PMTRIntraParam : public MTRIntraWater<TIP3PMTRIntraParam>
    {
       private:
        // clang-format off
        static constexpr auto _eqOHDistance =    0.9572;          // Angström
        static constexpr auto _eqHHDistance =    1.5139;          // Angström
        static constexpr auto _DOH          =  101.9048757170172; // kcal mol^-1
        static constexpr auto _alpha        =    2.483;           // Angström^-1
        static constexpr auto _beta         =    3.0;             // Angström^-2
        static constexpr auto _Ltt          =  235.2449808795411; // kcal mol^-1 Angström^-2 
        static constexpr auto _Lrt          = -181.2906309751434; // kcal mol^-1 Angström^-2
        static constexpr auto _Lrr          =  127.1534416826004; // kcal mol^-1 Angström^-2
        // clang-format on

        friend class MTRIntraWater<TIP3PMTRIntraParam>;
    };

}   // namespace waterModel

#include "mTRIntraWater.tpp.hpp"   // DO NOT MOVE THIS LINE

#endif   //  _MTR_INTRA_WATER_HPP_