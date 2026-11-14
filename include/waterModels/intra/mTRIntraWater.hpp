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

#ifndef WATER_MODELS__INTRA__M_TR_INTRA_WATER_HPP
#define WATER_MODELS__INTRA__M_TR_INTRA_WATER_HPP

#include "intraWater.hpp"
#include "unit_types.hpp"

namespace waterModel
{
    class MTRIntraWater : public IntraWater
    {
       public:
        void calculate(
            molsys::SimulationBox      &simulationBox,
            physicalData::PhysicalData &physicalData
        ) override;

        [[nodiscard]] virtual Distance    getEqOHDistance() const = 0;
        [[nodiscard]] virtual Distance    getEqHHDistance() const = 0;
        [[nodiscard]] virtual double      getDOH() const   = 0;   // kcal mol^-1
        [[nodiscard]] virtual InvDistance getAlpha() const = 0;
        [[nodiscard]] virtual InvDistanceSquared getBeta() const = 0;
        [[nodiscard]]
        virtual double getLtt() const = 0;   // kcal mol^-1 Angström^-2
        [[nodiscard]]
        virtual double getLrt() const = 0;   // kcal mol^-1 Angström^-2
        [[nodiscard]]
        virtual double getLrr() const = 0;   // kcal mol^-1 Angström^-2
    };

    class SPCMTRIntraWater : public MTRIntraWater
    {
       private:
        static constexpr Distance _eqOHDistance = 1.0 * mstd::units::angstrom;
        static constexpr Distance _eqHHDistance =
            1.632993162 * mstd::units::angstrom;
        static constexpr double      _dOH = 101.9048757170172;   // kcal mol^-1
        static constexpr InvDistance _alpha = 2.511 / mstd::units::angstrom;
        static constexpr InvDistanceSquared _beta =
            3.0 / (mstd::units::angstrom * mstd::units::angstrom);
        static constexpr double _ltt =
            264.5841300191204;   // kcal mol^-1 Angström^-2
        static constexpr double _lrt =
            -211.0444550669216;   // kcal mol^-1 Angström^-2
        static constexpr double _lrr =
            155.7839388145315;   // kcal mol^-1 Angström^-2

       public:
        [[nodiscard]]
        Distance getEqOHDistance() const override
        {
            return _eqOHDistance;
        }
        [[nodiscard]]
        Distance getEqHHDistance() const override
        {
            return _eqHHDistance;
        }
        [[nodiscard]] double      getDOH() const override { return _dOH; }
        [[nodiscard]] InvDistance getAlpha() const override { return _alpha; }
        [[nodiscard]] InvDistanceSquared getBeta() const override
        {
            return _beta;
        }
        [[nodiscard]] double getLtt() const override { return _ltt; }
        [[nodiscard]] double getLrt() const override { return _lrt; }
        [[nodiscard]] double getLrr() const override { return _lrr; }
    };

    class TIP3PMTRIntraWater : public MTRIntraWater
    {
       private:
        static constexpr Distance _eqOHDistance =
            0.9572 * mstd::units::angstrom;
        static constexpr Distance _eqHHDistance =
            1.5139 * mstd::units::angstrom;
        static constexpr double      _dOH = 101.9048757170172;   // kcal mol^-1
        static constexpr InvDistance _alpha = 2.483 / mstd::units::angstrom;
        static constexpr InvDistanceSquared _beta =
            3.0 / (mstd::units::angstrom * mstd::units::angstrom);
        static constexpr double _ltt =
            235.2449808795411;   // kcal mol^-1 Angström^-2
        static constexpr double _lrt =
            -181.2906309751434;   // kcal mol^-1 Angström^-2
        static constexpr double _lrr =
            127.1534416826004;   // kcal mol^-1 Angström^-2

       public:
        [[nodiscard]]
        Distance getEqOHDistance() const override
        {
            return _eqOHDistance;
        }
        [[nodiscard]]
        Distance getEqHHDistance() const override
        {
            return _eqHHDistance;
        }
        [[nodiscard]] double      getDOH() const override { return _dOH; }
        [[nodiscard]] InvDistance getAlpha() const override { return _alpha; }
        [[nodiscard]] InvDistanceSquared getBeta() const override
        {
            return _beta;
        }
        [[nodiscard]] double getLtt() const override { return _ltt; }
        [[nodiscard]] double getLrt() const override { return _lrt; }
        [[nodiscard]] double getLrr() const override { return _lrr; }
    };

}   // namespace waterModel

#endif   //  WATER_MODELS__INTRA__M_TR_INTRA_WATER_HPP
