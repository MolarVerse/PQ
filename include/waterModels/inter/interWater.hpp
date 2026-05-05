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
        double _chargeProductOO{};
        double _chargeProductOH{};
        double _chargeProductHH{};
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
        virtual ~InterWaterStrategy() = default;

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
        void initChargeProducts();
        void initState()
        {
            initGuffPairs();
            initChargeProducts();
        }
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

}   // namespace waterModel

#include "interWaterParamters.hpp"   // DO NOT MOVE THIS LINE

#endif   //  _INTER_WATER_HPP_