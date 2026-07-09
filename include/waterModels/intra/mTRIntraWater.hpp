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
    struct MTRIntraWaterParam
    {
        double eqOHDistance = 0.0;   // Angström
        double eqHHDistance = 0.0;   // Angström
        double DOH          = 0.0;   // kcal mol^-1
        double alpha        = 0.0;   // Angström^-1
        double beta         = 0.0;   // Angström^-2
        double Ltt          = 0.0;   // kcal mol^-1 Angström^-2
        double Lrt          = 0.0;   // kcal mol^-1 Angström^-2
        double Lrr          = 0.0;   // kcal mol^-1 Angström^-2
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
        MTRIntraWaterParam _parameters;

       public:
        SPCMTRIntraWater()
        {
            _parameters.eqOHDistance = 1.0;                 // Angström
            _parameters.eqHHDistance = 1.632993162;         // Angström
            _parameters.DOH          = 101.9048757170172;   // kcal mol^-1
            _parameters.alpha        = 2.511;               // Angström^-1
            _parameters.beta         = 3.0;                 // Angström^-2
            _parameters.Ltt = 264.5841300191204;    // kcal mol^-1 Angström^-2
            _parameters.Lrt = -211.0444550669216;   // kcal mol^-1 Angström^-2
            _parameters.Lrr = 155.7839388145315;    // kcal mol^-1 Angström^-2
        }

        const MTRIntraWaterParam &get_parameters() const final
        {
            return _parameters;
        }
    };

    class TIP3PMTRIntraWater : public MTRIntraWater
    {
       private:
        MTRIntraWaterParam _parameters;

       public:
        TIP3PMTRIntraWater()
        {
            _parameters.eqOHDistance = 0.9572;              // Angström
            _parameters.eqHHDistance = 1.5139;              // Angström
            _parameters.DOH          = 101.9048757170172;   // kcal mol^-1
            _parameters.alpha        = 2.483;               // Angström^-1
            _parameters.beta         = 3.0;                 // Angström^-2
            _parameters.Ltt = 235.2449808795411;    // kcal mol^-1 Angström^-2
            _parameters.Lrt = -181.2906309751434;   // kcal mol^-1 Angström^-2
            _parameters.Lrr = 127.1534416826004;    // kcal mol^-1 Angström^-2
        }

        const MTRIntraWaterParam &get_parameters() const final
        {
            return _parameters;
        }
    };

}   // namespace waterModel

#include "mTRIntraWater.tpp.hpp"   // DO NOT MOVE THIS LINE

#endif   //  _MTR_INTRA_WATER_HPP_