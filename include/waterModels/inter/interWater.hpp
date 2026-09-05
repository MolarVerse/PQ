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

#include <memory>

#include "coulombPotential.hpp"
#include "nonCoulombPair.hpp"

namespace molsys
{
    class SimulationBox;   // forward declaration
    class CellList;        // forward declaration
    class Atom;            // forward declaration
}   // namespace molsys

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

        std::unique_ptr<pot::NonCoulombPair> _nonCoulombPairOO;
        std::unique_ptr<pot::NonCoulombPair> _nonCoulombPairOH;
        std::unique_ptr<pot::NonCoulombPair> _nonCoulombPairHH;
        // clang-format on
    };

    class InterWaterStrategy
    {
       public:
        virtual ~InterWaterStrategy() = default;

        virtual void calculate(
            const InterWaterState &,
            molsys::SimulationBox &,
            physicalData::PhysicalData &,
            const std::shared_ptr<pot::CoulombPotential> &,
            molsys::CellList &
        ) = 0;

        virtual void calculateCoreToOuterForces(
            const InterWaterState &,
            molsys::SimulationBox &,
            physicalData::PhysicalData &,
            const std::shared_ptr<pot::CoulombPotential> &,
            molsys::CellList &
        ) = 0;

        virtual void calculateLayerToOuterForces(
            const InterWaterState &,
            molsys::SimulationBox &,
            physicalData::PhysicalData &,
            const std::shared_ptr<pot::CoulombPotential> &,
            molsys::CellList &
        ) = 0;

        virtual void calculateOuterToOuterForces(
            const InterWaterState &,
            molsys::SimulationBox &,
            physicalData::PhysicalData &,
            const std::shared_ptr<pot::CoulombPotential> &,
            molsys::CellList &
        ) = 0;

        virtual void calculateHotspotSmoothingMMForces(
            const InterWaterState &,
            molsys::SimulationBox &,
            physicalData::PhysicalData &,
            const std::shared_ptr<pot::CoulombPotential> &,
            molsys::CellList &
        ) = 0;

        template <typename ChargeTag1, typename ChargeTag2>
        void calculateSingleInteraction(
            molsys::Atom                                 &atom1,
            molsys::Atom                                 &atom2,
            const std::shared_ptr<pot::CoulombPotential> &coulombPotential,
            const double                                  rCutSquared,
            const molsys::SimulationBox                  &simBox,
            const pot::NonCoulombPair                    &nonCoulPair,
            double                                       &coulombEnergy,
            double                                       &nonCoulombEnergy
        );

        template <typename ChargeTag1, typename ChargeTag2>
        void calculateSingleCoulombInteraction(
            molsys::Atom                                 &atom1,
            molsys::Atom                                 &atom2,
            const std::shared_ptr<pot::CoulombPotential> &coulombPotential,
            const double                                  rCutSquared,
            const molsys::SimulationBox                  &simBox,
            double                                       &coulombEnergy
        );

        template <typename ChargeTag1, typename ChargeTag2>
        void calculateSingleInteractionOneWay(
            molsys::Atom                                 &atom1,
            molsys::Atom                                 &atom2,
            const std::shared_ptr<pot::CoulombPotential> &coulombPotential,
            const double                                  rCutSquared,
            const molsys::SimulationBox                  &simBox,
            const pot::NonCoulombPair                    &nonCoulPair,
            double                                       &coulombEnergy,
            double                                       &nonCoulombEnergy
        );

        template <typename T>
        double getPartialCharge(molsys::Atom &atom) const;
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
            molsys::SimulationBox                        &simBox,
            physicalData::PhysicalData                   &physicalData,
            const std::shared_ptr<pot::CoulombPotential> &sharedCoulombPot,
            molsys::CellList                             &cellList
        );

        void calculateQMMMForces(
            molsys::SimulationBox                        &simBox,
            physicalData::PhysicalData                   &physicalData,
            const std::shared_ptr<pot::CoulombPotential> &sharedCoulombPot,
            molsys::CellList                             &cellList
        );

        void calculateHotspotSmoothingMMForces(
            molsys::SimulationBox                        &simBox,
            physicalData::PhysicalData                   &physicalData,
            const std::shared_ptr<pot::CoulombPotential> &sharedCoulombPot,
            molsys::CellList                             &cellList
        );

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
        void calculate(
            const InterWaterState & /*interWaterState*/,
            molsys::SimulationBox & /*simBox*/,
            physicalData::PhysicalData & /*physData*/,
            const std::shared_ptr<pot::CoulombPotential> & /*coulPot*/,
            molsys::CellList & /*cellList*/
        ) final
        {
        }

        void calculateCoreToOuterForces(
            const InterWaterState & /*interWaterState*/,
            molsys::SimulationBox & /*simBox*/,
            physicalData::PhysicalData & /*physData*/,
            const std::shared_ptr<pot::CoulombPotential> & /*coulPot*/,
            molsys::CellList & /*cellList*/
        ) final
        {
        }

        void calculateLayerToOuterForces(
            const InterWaterState & /*interWaterState*/,
            molsys::SimulationBox & /*simBox*/,
            physicalData::PhysicalData & /*physData*/,
            const std::shared_ptr<pot::CoulombPotential> & /*coulPot*/,
            molsys::CellList & /*cellList*/
        ) final
        {
        }

        void calculateOuterToOuterForces(
            const InterWaterState & /*interWaterState*/,
            molsys::SimulationBox & /*simBox*/,
            physicalData::PhysicalData & /*physData*/,
            const std::shared_ptr<pot::CoulombPotential> & /*coulPot*/,
            molsys::CellList & /*cellList*/
        ) final
        {
        }

        void calculateHotspotSmoothingMMForces(
            const InterWaterState & /*interWaterState*/,
            molsys::SimulationBox & /*simBox*/,
            physicalData::PhysicalData & /*physData*/,
            const std::shared_ptr<pot::CoulombPotential> & /*coulPot*/,
            molsys::CellList & /*cellList*/
        ) final
        {
        }
    };

    class InterWaterStrategyBruteForce : public InterWaterStrategy
    {
       public:
        void calculate(
            const InterWaterState                        &interWaterState,
            molsys::SimulationBox                        &simBox,
            physicalData::PhysicalData                   &physData,
            const std::shared_ptr<pot::CoulombPotential> &coulPot,
            molsys::CellList                             &cellList
        ) final;

        void calculateCoreToOuterForces(
            const InterWaterState                        &interWaterState,
            molsys::SimulationBox                        &simBox,
            physicalData::PhysicalData                   &physData,
            const std::shared_ptr<pot::CoulombPotential> &coulPot,
            molsys::CellList                             &cellList
        ) final;

        void calculateLayerToOuterForces(
            const InterWaterState                        &interWaterState,
            molsys::SimulationBox                        &simBox,
            physicalData::PhysicalData                   &physData,
            const std::shared_ptr<pot::CoulombPotential> &coulPot,
            molsys::CellList                             &cellList
        ) final;

        void calculateOuterToOuterForces(
            const InterWaterState                        &interWaterState,
            molsys::SimulationBox                        &simBox,
            physicalData::PhysicalData                   &physData,
            const std::shared_ptr<pot::CoulombPotential> &coulPot,
            molsys::CellList                             &cellList
        ) final;

        void calculateHotspotSmoothingMMForces(
            const InterWaterState                        &interWaterState,
            molsys::SimulationBox                        &simBox,
            physicalData::PhysicalData                   &physData,
            const std::shared_ptr<pot::CoulombPotential> &coulPot,
            molsys::CellList                             &cellList
        ) final;
    };

    class InterWaterStrategyCellList : public InterWaterStrategy
    {
       public:
        void calculate(
            const InterWaterState                        &interWaterState,
            molsys::SimulationBox                        &simBox,
            physicalData::PhysicalData                   &physData,
            const std::shared_ptr<pot::CoulombPotential> &coulPot,
            molsys::CellList                             &cellList
        ) final;

        void calculateCoreToOuterForces(
            const InterWaterState                        &interWaterState,
            molsys::SimulationBox                        &simBox,
            physicalData::PhysicalData                   &physData,
            const std::shared_ptr<pot::CoulombPotential> &coulPot,
            molsys::CellList                             &cellList
        ) final;

        void calculateLayerToOuterForces(
            const InterWaterState                        &interWaterState,
            molsys::SimulationBox                        &simBox,
            physicalData::PhysicalData                   &physData,
            const std::shared_ptr<pot::CoulombPotential> &coulPot,
            molsys::CellList                             &cellList
        ) final;

        void calculateOuterToOuterForces(
            const InterWaterState                        &interWaterState,
            molsys::SimulationBox                        &simBox,
            physicalData::PhysicalData                   &physData,
            const std::shared_ptr<pot::CoulombPotential> &coulPot,
            molsys::CellList                             &cellList
        ) final;

        void calculateHotspotSmoothingMMForces(
            const InterWaterState                        &interWaterState,
            molsys::SimulationBox                        &simBox,
            physicalData::PhysicalData                   &physData,
            const std::shared_ptr<pot::CoulombPotential> &coulPot,
            molsys::CellList                             &cellList
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
