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

#ifndef WATER_MODELS__INTRA__SPC_INTRA_WATER_HPP
#define WATER_MODELS__INTRA__SPC_INTRA_WATER_HPP

#include "intraWater.hpp"
#include "unit_types.hpp"

namespace molsys
{
    class SimulationBox;   // forward declaration
}   // namespace molsys

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
            molsys::SimulationBox      &simulationBox,
            physicalData::PhysicalData &physicalData
        ) override;

        [[nodiscard]]
        virtual const Distance &getEqOHDistance() const = 0;
        [[nodiscard]]
        virtual const Angle &getEqHOHAngle() const = 0;
        [[nodiscard]]
        virtual const ForceConstant &getForceConstantOHBond() const = 0;
        [[nodiscard]]
        virtual const AngleForceConstant &getForceConstantHOHAngle() const = 0;
    };

    class SPCFwIntraWater : public SPCIntraWater
    {
       private:
        static constexpr Distance _eqOHDistance = 1.012 * mstd::units::angstrom;
        static constexpr Angle    _eqHOHAngle   = 113.24 * mstd::units::deg;
        static constexpr ForceConstant _forceConstantOHBond =
            1059.162 * mstd::units::kcal_per_mol /
            (mstd::units::angstrom * mstd::units::angstrom);
        static constexpr AngleForceConstant _forceConstantHOHAngle =
            75.9 * mstd::units::kcal_per_mol /
            (mstd::units::rad * mstd::units::rad);

       public:
        [[nodiscard]]
        const Distance &getEqOHDistance() const override
        {
            return _eqOHDistance;
        }
        [[nodiscard]]
        const Angle &getEqHOHAngle() const override
        {
            return _eqHOHAngle;
        }
        [[nodiscard]]
        const ForceConstant &getForceConstantOHBond() const override
        {
            return _forceConstantOHBond;
        }
        [[nodiscard]]
        const AngleForceConstant &getForceConstantHOHAngle() const override
        {
            return _forceConstantHOHAngle;
        }
    };

    class qSPCFwIntraWater : public SPCIntraWater
    {
       private:
        static constexpr Distance _eqOHDistance = 1.0 * mstd::units::angstrom;
        static constexpr Angle    _eqHOHAngle   = 112.0 * mstd::units::deg;
        static constexpr ForceConstant _forceConstantOHBond =
            1059.162 * mstd::units::kcal_per_mol /
            (mstd::units::angstrom * mstd::units::angstrom);
        static constexpr AngleForceConstant _forceConstantHOHAngle =
            75.9 * mstd::units::kcal_per_mol /
            (mstd::units::rad * mstd::units::rad);

       public:
        [[nodiscard]]
        const Distance &getEqOHDistance() const override
        {
            return _eqOHDistance;
        }
        [[nodiscard]]
        const Angle &getEqHOHAngle() const override
        {
            return _eqHOHAngle;
        }
        [[nodiscard]]
        const ForceConstant &getForceConstantOHBond() const override
        {
            return _forceConstantOHBond;
        }
        [[nodiscard]]
        const AngleForceConstant &getForceConstantHOHAngle() const override
        {
            return _forceConstantHOHAngle;
        }
    };

}   // namespace waterModel

#ifndef WATER_MODELS__INTRA__SPC_INTRA_WATER_TPP
#include "SPCIntraWater.tpp"   // IWYU pragma: export
#endif

#endif   //  WATER_MODELS__INTRA__SPC_INTRA_WATER_HPP
