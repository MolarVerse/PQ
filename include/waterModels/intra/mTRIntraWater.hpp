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

#include <mstd/units/new_units.hpp>

#include "intraWater.hpp"
#include "unit_types.hpp"

namespace waterModel
{
    class MTRIntraWater : public IntraWater
    {
       public:
        using IntraWater::IntraWater;
        void calculate(
            molsys::SimulationBox      &simulationBox,
            physicalData::PhysicalData &physicalData
        ) override;

        [[nodiscard]] virtual Distance           getEqOHDistance() const = 0;
        [[nodiscard]] virtual Distance           getEqHHDistance() const = 0;
        [[nodiscard]] virtual EnergyPerMol       getDOH() const          = 0;
        [[nodiscard]] virtual InvDistance        getAlpha() const        = 0;
        [[nodiscard]] virtual InvDistanceSquared getBeta() const         = 0;
        [[nodiscard]]
        virtual ForceConstant getLtt() const = 0;
        [[nodiscard]]
        virtual ForceConstant getLrt() const = 0;
        [[nodiscard]]
        virtual ForceConstant getLrr() const = 0;
    };

    class SPCMTRIntraWater : public MTRIntraWater
    {
       private:
        static constexpr Distance _eqOHDistance = 1.0 * mstd::units::angstrom;
        static constexpr Distance _eqHHDistance =
            1.632993162 * mstd::units::angstrom;
        static constexpr EnergyPerMol _dOH =
            101.9048757170172 * mstd::units::kcal_per_mol;
        static constexpr InvDistance _alpha = 2.511 / mstd::units::angstrom;
        static constexpr InvDistanceSquared _beta =
            3.0 / (mstd::units::angstrom * mstd::units::angstrom);
        static constexpr ForceConstant _ltt =
            264.5841300191204 * mstd::units::kcal_per_mol /
            (mstd::units::angstrom * mstd::units::angstrom);
        static constexpr ForceConstant _lrt =
            -211.0444550669216 * mstd::units::kcal_per_mol /
            (mstd::units::angstrom * mstd::units::angstrom);
        static constexpr ForceConstant _lrr =
            155.7839388145315 * mstd::units::kcal_per_mol /
            (mstd::units::angstrom * mstd::units::angstrom);

       public:
        using MTRIntraWater::MTRIntraWater;

        [[nodiscard]] Distance getEqOHDistance() const override;
        [[nodiscard]] Distance getEqHHDistance() const override;
        [[nodiscard]] EnergyPerMol getDOH() const override;
        [[nodiscard]] InvDistance getAlpha() const override;
        [[nodiscard]] InvDistanceSquared getBeta() const override;
        [[nodiscard]] ForceConstant getLtt() const override;
        [[nodiscard]] ForceConstant getLrt() const override;
        [[nodiscard]] ForceConstant getLrr() const override;
    };

    inline Distance SPCMTRIntraWater::getEqOHDistance() const
    {
        return _eqOHDistance;
    }

    inline Distance SPCMTRIntraWater::getEqHHDistance() const
    {
        return _eqHHDistance;
    }

    inline EnergyPerMol SPCMTRIntraWater::getDOH() const { return _dOH; }

    inline InvDistance SPCMTRIntraWater::getAlpha() const { return _alpha; }

    inline InvDistanceSquared SPCMTRIntraWater::getBeta() const { return _beta; }

    inline ForceConstant SPCMTRIntraWater::getLtt() const { return _ltt; }

    inline ForceConstant SPCMTRIntraWater::getLrt() const { return _lrt; }

    inline ForceConstant SPCMTRIntraWater::getLrr() const { return _lrr; }

    class TIP3PMTRIntraWater : public MTRIntraWater
    {
       private:
        static constexpr Distance _eqOHDistance =
            0.9572 * mstd::units::angstrom;
        static constexpr Distance _eqHHDistance =
            1.5139 * mstd::units::angstrom;
        static constexpr EnergyPerMol _dOH =
            101.9048757170172 * mstd::units::kcal_per_mol;
        static constexpr InvDistance _alpha = 2.483 / mstd::units::angstrom;
        static constexpr InvDistanceSquared _beta =
            3.0 / (mstd::units::angstrom * mstd::units::angstrom);
        static constexpr ForceConstant _ltt =
            235.2449808795411 * mstd::units::kcal_per_mol /
            (mstd::units::angstrom * mstd::units::angstrom);
        static constexpr ForceConstant _lrt =
            -181.2906309751434 * mstd::units::kcal_per_mol /
            (mstd::units::angstrom * mstd::units::angstrom);
        static constexpr ForceConstant _lrr =
            127.1534416826004 * mstd::units::kcal_per_mol /
            (mstd::units::angstrom * mstd::units::angstrom);

       public:
        using MTRIntraWater::MTRIntraWater;
        [[nodiscard]] Distance getEqOHDistance() const override;
        [[nodiscard]] Distance getEqHHDistance() const override;
        [[nodiscard]] EnergyPerMol getDOH() const override;
        [[nodiscard]] InvDistance getAlpha() const override;
        [[nodiscard]] InvDistanceSquared getBeta() const override;
        [[nodiscard]] ForceConstant getLtt() const override;
        [[nodiscard]] ForceConstant getLrt() const override;
        [[nodiscard]] ForceConstant getLrr() const override;
    };

    inline Distance TIP3PMTRIntraWater::getEqOHDistance() const
    {
        return _eqOHDistance;
    }

    inline Distance TIP3PMTRIntraWater::getEqHHDistance() const
    {
        return _eqHHDistance;
    }

    inline EnergyPerMol TIP3PMTRIntraWater::getDOH() const { return _dOH; }

    inline InvDistance TIP3PMTRIntraWater::getAlpha() const { return _alpha; }

    inline InvDistanceSquared TIP3PMTRIntraWater::getBeta() const { return _beta; }

    inline ForceConstant TIP3PMTRIntraWater::getLtt() const { return _ltt; }

    inline ForceConstant TIP3PMTRIntraWater::getLrt() const { return _lrt; }

    inline ForceConstant TIP3PMTRIntraWater::getLrr() const { return _lrr; }

}   // namespace waterModel

#endif   //  WATER_MODELS__INTRA__M_TR_INTRA_WATER_HPP
