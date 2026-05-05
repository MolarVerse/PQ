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

#ifndef _INTER_WATER_HPP_

#define _INTER_WATER_HPP_

#include <utility>
#include <vector>

#include "atom.hpp"                // for Atom
#include "defaults.hpp"            // for defaults
#include "guffCoefficients.hpp"    // for guffPairCoefficients
#include "guffPair.hpp"            // for GuffPair
#include "physicalData.hpp"        // for PhysicalData
#include "potentialSettings.hpp"   // for PotentialSettings
#include "simulationBox.hpp"       // for SimulationBox
#include "typeAliases.hpp"

namespace waterModel
{
    struct InterWaterState
    {
        double _oxygenCharge{};
        double _hydrogenCharge{};
        double _nonCoulombCutOff = defaults::_COULOMB_CUT_OFF_DEFAULT_;
        std::vector<double> _guffCoefficientsOO;
        std::vector<double> _guffCoefficientsOH;
        std::vector<double> _guffCoefficientsHH;
        potential::GuffPair _guffPairOO{_nonCoulombCutOff, _guffCoefficientsOO};
        potential::GuffPair _guffPairOH{_nonCoulombCutOff, _guffCoefficientsOH};
        potential::GuffPair _guffPairHH{_nonCoulombCutOff, _guffCoefficientsHH};
    };

    class InterWaterStrategy
    {
       public:
        ~InterWaterStrategy() = default;

        virtual void calculate(
            const InterWaterState &,
            pq::SimBox &,
            pq::PhysicalData &,
            const pq::SharedCoulombPot &
        ) = 0;

        std::pair<double, double> calculateSingleInteraction(
            pq::Atom                   &atom1,
            pq::Atom                   &atom2,
            const double                chargeProduct,
            const pq::SharedCoulombPot &coulombPotential,
            const double                rCutSquared,
            const pq::SimBox           &simBox,
            const potential::GuffPair  &guffPair
        );
    };

    class InterWater
    {
       public:
        InterWater();

        InterWater(
            InterWaterState                     state,
            std::unique_ptr<InterWaterStrategy> strategy
        );

        void calculate(
            pq::SimBox                 &simBox,
            pq::PhysicalData           &physicalData,
            const pq::SharedCoulombPot &sharedCoulombPot
        )
        {
            if (!_strategy)
                return;

            _strategy
                ->calculate(_state, simBox, physicalData, sharedCoulombPot);
        }

       private:
        InterWaterState                     _state;
        std::unique_ptr<InterWaterStrategy> _strategy;

        void initGuffPairs();
    };

    class InterWaterStrategyNull : public InterWaterStrategy
    {
       public:
        virtual void calculate(
            const InterWaterState &,
            pq::SimBox &,
            pq::PhysicalData &,
            const pq::SharedCoulombPot &
        ) override final
        {
        }
    };

    class InterWaterStrategyBruteForce : public InterWaterStrategy
    {
       public:
        virtual void calculate(
            const InterWaterState &,
            pq::SimBox &,
            pq::PhysicalData &,
            const pq::SharedCoulombPot &
        ) override final;
    };

    struct SPCFwInterParam
    {
        static constexpr auto                   _oxygenCharge   = -0.82;
        static constexpr auto                   _hydrogenCharge = 0.41;
        inline static const std::vector<double> _guffCoefficientsOO =
            constants::_SPC_FW_GUFF_COEFFICIENTS_OO_;
        inline static const std::vector<double> _guffCoefficientsOH =
            constants::_ZERO_GUFF_COEFFICIENTS_;
        inline static const std::vector<double> _guffCoefficientsHH =
            constants::_ZERO_GUFF_COEFFICIENTS_;
    };

    struct qSPCFwInterParam
    {
        static constexpr auto                   _oxygenCharge   = -0.84;
        static constexpr auto                   _hydrogenCharge = 0.42;
        inline static const std::vector<double> _guffCoefficientsOO =
            constants::_QSPC_FW_GUFF_COEFFICIENTS_OO_;
        inline static const std::vector<double> _guffCoefficientsOH =
            constants::_ZERO_GUFF_COEFFICIENTS_;
        inline static const std::vector<double> _guffCoefficientsHH =
            constants::_ZERO_GUFF_COEFFICIENTS_;
    };

    struct TIP3PInterParam
    {
        static constexpr auto                   _oxygenCharge   = -0.834;
        static constexpr auto                   _hydrogenCharge = 0.417;
        inline static const std::vector<double> _guffCoefficientsOO =
            constants::_TIP3P_GUFF_COEFFICIENTS_OO_;
        inline static const std::vector<double> _guffCoefficientsOH =
            constants::_ZERO_GUFF_COEFFICIENTS_;
        inline static const std::vector<double> _guffCoefficientsHH =
            constants::_ZERO_GUFF_COEFFICIENTS_;
    };

    struct OPC3InterParam
    {
        static constexpr auto                   _oxygenCharge   = -0.89517;
        static constexpr auto                   _hydrogenCharge = 0.447585;
        inline static const std::vector<double> _guffCoefficientsOO =
            constants::_OPC3_GUFF_COEFFICIENTS_OO_;
        inline static const std::vector<double> _guffCoefficientsOH =
            constants::_ZERO_GUFF_COEFFICIENTS_;
        inline static const std::vector<double> _guffCoefficientsHH =
            constants::_ZERO_GUFF_COEFFICIENTS_;
    };

    template <class T>
    concept InterWaterParameterClass = requires {
        { T::_oxygenCharge } -> std::convertible_to<double>;
        { T::_hydrogenCharge } -> std::convertible_to<double>;
        { T::_guffCoefficientsOO } -> std::convertible_to<std::vector<double>>;
        { T::_guffCoefficientsOH } -> std::convertible_to<std::vector<double>>;
        { T::_guffCoefficientsHH } -> std::convertible_to<std::vector<double>>;
    };

    template <InterWaterParameterClass T>
    inline InterWaterState makeInterWaterState()
    {
        auto state                = InterWaterState();
        state._oxygenCharge       = T::_oxygenCharge;
        state._hydrogenCharge     = T::_hydrogenCharge;
        state._guffCoefficientsOO = T::_guffCoefficientsOO;
        state._guffCoefficientsOH = T::_guffCoefficientsOH;
        state._guffCoefficientsHH = T::_guffCoefficientsHH;

        return state;
    }

}   // namespace waterModel

#include "interWater.tpp.hpp"   // DO NOT MOVE THIS LINE

#endif   //  _INTER_WATER_HPP_