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

#include "coulombPotential.hpp"
#include "nonCoulombPair.hpp"
#include "timer.hpp"

namespace simulationBox
{
    class SimulationBox;   // forward declaration
    class CellList;        // forward declaration
    class Atom;            // forward declaration
}   // namespace simulationBox

namespace physicalData
{
    class PhysicalData;   // forward declaration
}   // namespace physicalData

namespace waterModel
{
    struct InterWaterState
    {
        // clang-format off
        double _oxygenCharge{};
        double _hydrogenCharge{};
        bool   _oxygenOnlyNonCoulomb{false};

        std::unique_ptr<potential::NonCoulombPair> _nonCoulombPairOO;
        std::unique_ptr<potential::NonCoulombPair> _nonCoulombPairOH;
        std::unique_ptr<potential::NonCoulombPair> _nonCoulombPairHH;
        // clang-format on
    };

    class InterWaterStrategy
    {
       public:
        virtual ~InterWaterStrategy() = default;

        virtual void calculate(
            const InterWaterState &,
            simulationBox::SimulationBox &,
            physicalData::PhysicalData &,
            const std::shared_ptr<potential::CoulombPotential> &,
            simulationBox::CellList &
        ) = 0;

        virtual void calculateCoreToOuterForces(
            const InterWaterState &,
            simulationBox::SimulationBox &,
            physicalData::PhysicalData &,
            const std::shared_ptr<potential::CoulombPotential> &,
            simulationBox::CellList &
        ) = 0;

        virtual void calculateLayerToOuterForces(
            const InterWaterState &,
            simulationBox::SimulationBox &,
            physicalData::PhysicalData &,
            const std::shared_ptr<potential::CoulombPotential> &,
            simulationBox::CellList &
        ) = 0;

        virtual void calculateOuterToOuterForces(
            const InterWaterState &,
            simulationBox::SimulationBox &,
            physicalData::PhysicalData &,
            const std::shared_ptr<potential::CoulombPotential> &,
            simulationBox::CellList &
        ) = 0;

        virtual void calculateHotspotSmoothingMMForces(
            const InterWaterState &,
            simulationBox::SimulationBox &,
            physicalData::PhysicalData &,
            const std::shared_ptr<potential::CoulombPotential> &,
            simulationBox::CellList &
        ) = 0;

        template <typename ChargeTag1, typename ChargeTag2>
        void calculateSingleInteraction(
            simulationBox::Atom &atom1,
            simulationBox::Atom &atom2,
            const std::shared_ptr<potential::CoulombPotential>
                                               &coulombPotential,
            const double                        rCutSquared,
            const simulationBox::SimulationBox &simBox,
            const potential::NonCoulombPair    &nonCoulPair,
            double                             &coulombEnergy,
            double                             &nonCoulombEnergy
        );

        template <typename ChargeTag1, typename ChargeTag2>
        void calculateSingleCoulombInteraction(
            simulationBox::Atom &atom1,
            simulationBox::Atom &atom2,
            const std::shared_ptr<potential::CoulombPotential>
                                               &coulombPotential,
            const double                        rCutSquared,
            const simulationBox::SimulationBox &simBox,
            double                             &coulombEnergy
        );

        template <typename ChargeTag1, typename ChargeTag2>
        void calculateSingleInteractionOneWay(
            simulationBox::Atom &atom1,
            simulationBox::Atom &atom2,
            const std::shared_ptr<potential::CoulombPotential>
                                               &coulombPotential,
            const double                        rCutSquared,
            const simulationBox::SimulationBox &simBox,
            const potential::NonCoulombPair    &nonCoulPair,
            double                             &coulombEnergy,
            double                             &nonCoulombEnergy
        );

        template <typename T>
        double getPartialCharge(simulationBox::Atom &atom) const;
    };

    class InterWater : public timings::Timer
    {
       public:
        InterWater();

        InterWater(
            InterWaterState                     state,
            std::unique_ptr<InterWaterStrategy> strategy
        );

        void calculate(
            simulationBox::SimulationBox &simBox,
            physicalData::PhysicalData   &physicalData,
            const std::shared_ptr<potential::CoulombPotential>
                                    &sharedCoulombPot,
            simulationBox::CellList &cellList
        );

        void calculateQMMMForces(
            simulationBox::SimulationBox &simBox,
            physicalData::PhysicalData   &physicalData,
            const std::shared_ptr<potential::CoulombPotential>
                                    &sharedCoulombPot,
            simulationBox::CellList &cellList
        );

        void calculateHotspotSmoothingMMForces(
            simulationBox::SimulationBox &simBox,
            physicalData::PhysicalData   &physicalData,
            const std::shared_ptr<potential::CoulombPotential>
                                    &sharedCoulombPot,
            simulationBox::CellList &cellList
        );

       private:
        InterWaterState                     _state;
        std::unique_ptr<InterWaterStrategy> _strategy;

        void addMTRSafetyPotential();
        void setNonCoulombCutOffRadii();
        void initNonCoulombPairs();
        void initState()
        {
            addMTRSafetyPotential();
            setNonCoulombCutOffRadii();
            initNonCoulombPairs();
        }
    };

    class InterWaterStrategyNull : public InterWaterStrategy
    {
       public:
        void calculate(
            const InterWaterState &,
            simulationBox::SimulationBox &,
            physicalData::PhysicalData &,
            const std::shared_ptr<potential::CoulombPotential> &,
            simulationBox::CellList &
        ) final
        {
        }

        void calculateCoreToOuterForces(
            const InterWaterState &,
            simulationBox::SimulationBox &,
            physicalData::PhysicalData &,
            const std::shared_ptr<potential::CoulombPotential> &,
            simulationBox::CellList &
        ) final
        {
        }

        void calculateLayerToOuterForces(
            const InterWaterState &,
            simulationBox::SimulationBox &,
            physicalData::PhysicalData &,
            const std::shared_ptr<potential::CoulombPotential> &,
            simulationBox::CellList &
        ) final
        {
        }

        void calculateOuterToOuterForces(
            const InterWaterState &,
            simulationBox::SimulationBox &,
            physicalData::PhysicalData &,
            const std::shared_ptr<potential::CoulombPotential> &,
            simulationBox::CellList &
        ) final
        {
        }

        void calculateHotspotSmoothingMMForces(
            const InterWaterState &,
            simulationBox::SimulationBox &,
            physicalData::PhysicalData &,
            const std::shared_ptr<potential::CoulombPotential> &,
            simulationBox::CellList &
        ) final
        {
        }
    };

    class InterWaterStrategyBruteForce : public InterWaterStrategy
    {
       public:
        void calculate(
            const InterWaterState &,
            simulationBox::SimulationBox &,
            physicalData::PhysicalData &,
            const std::shared_ptr<potential::CoulombPotential> &,
            simulationBox::CellList &
        ) final;

        void calculateCoreToOuterForces(
            const InterWaterState &,
            simulationBox::SimulationBox &,
            physicalData::PhysicalData &,
            const std::shared_ptr<potential::CoulombPotential> &,
            simulationBox::CellList &
        ) final;

        void calculateLayerToOuterForces(
            const InterWaterState &,
            simulationBox::SimulationBox &,
            physicalData::PhysicalData &,
            const std::shared_ptr<potential::CoulombPotential> &,
            simulationBox::CellList &
        ) final;

        void calculateOuterToOuterForces(
            const InterWaterState &,
            simulationBox::SimulationBox &,
            physicalData::PhysicalData &,
            const std::shared_ptr<potential::CoulombPotential> &,
            simulationBox::CellList &
        ) final;

        void calculateHotspotSmoothingMMForces(
            const InterWaterState &,
            simulationBox::SimulationBox &,
            physicalData::PhysicalData &,
            const std::shared_ptr<potential::CoulombPotential> &,
            simulationBox::CellList &
        ) final;
    };

    class InterWaterStrategyCellList : public InterWaterStrategy
    {
       public:
        void calculate(
            const InterWaterState &,
            simulationBox::SimulationBox &,
            physicalData::PhysicalData &,
            const std::shared_ptr<potential::CoulombPotential> &,
            simulationBox::CellList &
        ) final;

        void calculateCoreToOuterForces(
            const InterWaterState &,
            simulationBox::SimulationBox &,
            physicalData::PhysicalData &,
            const std::shared_ptr<potential::CoulombPotential> &,
            simulationBox::CellList &
        ) final;

        void calculateLayerToOuterForces(
            const InterWaterState &,
            simulationBox::SimulationBox &,
            physicalData::PhysicalData &,
            const std::shared_ptr<potential::CoulombPotential> &,
            simulationBox::CellList &
        ) final;

        void calculateOuterToOuterForces(
            const InterWaterState &,
            simulationBox::SimulationBox &,
            physicalData::PhysicalData &,
            const std::shared_ptr<potential::CoulombPotential> &,
            simulationBox::CellList &
        ) final;

        void calculateHotspotSmoothingMMForces(
            const InterWaterState &,
            simulationBox::SimulationBox &,
            physicalData::PhysicalData &,
            const std::shared_ptr<potential::CoulombPotential> &,
            simulationBox::CellList &
        ) final;
    };

}   // namespace waterModel

#ifndef _INTER_WATER_TPP_
#include "interWater.tpp"   // DO NOT MOVE THIS LINE
#endif

#ifndef _INTER_WATER_PARAMETERS_HPP_
#include "interWaterParamters.hpp"   // IWYU pragma: export - DO NOT MOVE THIS LINE
#endif

#endif   //  _INTER_WATER_HPP_
