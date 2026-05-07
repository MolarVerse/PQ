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
#include "celllist.hpp"            // for CellList
#include "defaults.hpp"            // for defaults
#include "guffCoefficients.hpp"    // for guffPairCoefficients
#include "nonCoulombPair.hpp"      // for NonCoulombPair
#include "physicalData.hpp"        // for PhysicalData
#include "potentialSettings.hpp"   // for PotentialSettings
#include "simulationBox.hpp"       // for SimulationBox
#include "typeAliases.hpp"

namespace waterModel
{
    struct InterWaterState
    {
        // clang-format off
        double _chargeProductOO{};
        double _chargeProductOH{};
        double _chargeProductHH{};
        bool   _oxygenOnlyNonCoulomb{false};
        std::unique_ptr<pq::NonCoulPair> _nonCoulombPairOO;
        std::unique_ptr<pq::NonCoulPair> _nonCoulombPairOH;
        std::unique_ptr<pq::NonCoulPair> _nonCoulombPairHH;
        // clang-format on
    };

    class InterWaterStrategy
    {
       public:
        virtual ~InterWaterStrategy() = default;

        virtual void calculate(
            const InterWaterState &,
            pq::SimBox &,
            pq::PhysicalData &,
            const pq::SharedCoulombPot &,
            pq::CellList &
        ) = 0;

        void calculateSingleInteraction(
            pq::Atom                   &atom1,
            pq::Atom                   &atom2,
            const double                chargeProduct,
            const pq::SharedCoulombPot &coulombPotential,
            const double                rCutSquared,
            const pq::SimBox           &simBox,
            const pq::NonCoulPair      &nonCoulPair,
            double                     &coulombEnergy,
            double                     &nonCoulombEnergy
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
            const pq::SharedCoulombPot &sharedCoulombPot,
            pq::CellList               &cellList
        )
        {
            if (!_strategy)
                return;

            _strategy->calculate(
                _state,
                simBox,
                physicalData,
                sharedCoulombPot,
                cellList
            );
        }

       private:
        InterWaterState                     _state;
        std::unique_ptr<InterWaterStrategy> _strategy;

        void setNonCoulombCutOffRadii();
        void initNonCoulombPairs();
        void initState()
        {
            setNonCoulombCutOffRadii();
            initNonCoulombPairs();
        }
    };

    class InterWaterStrategyNull : public InterWaterStrategy
    {
       public:
        virtual void calculate(
            const InterWaterState &,
            pq::SimBox &,
            pq::PhysicalData &,
            const pq::SharedCoulombPot &,
            pq::CellList &
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
            const pq::SharedCoulombPot &,
            pq::CellList &
        ) override final;
    };

    class InterWaterStrategyCellList : public InterWaterStrategy
    {
       public:
        virtual void calculate(
            const InterWaterState &,
            pq::SimBox &,
            pq::PhysicalData &,
            const pq::SharedCoulombPot &,
            pq::CellList &
        ) override final;
    };

}   // namespace waterModel

#include "interWaterParamters.hpp"   // DO NOT MOVE THIS LINE

#endif   //  _INTER_WATER_HPP_