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

#ifndef _MD_ENGINE_HPP_

#define _MD_ENGINE_HPP_

#include "boxOutput.hpp"
#include "energyOutput.hpp"
#include "engine.hpp"
#include "integrator.hpp"
#include "manostat.hpp"
#include "momentumOutput.hpp"
#include "resetKinetics.hpp"
#include "ringPolymerEnergyOutput.hpp"
#include "ringPolymerRestartFileOutput.hpp"
#include "ringPolymerTrajectoryOutput.hpp"
#include "stressOutput.hpp"
#include "thermostat.hpp"
#include "trajectoryOutput.hpp"
#include "virialOutput.hpp"

namespace engine
{
    /**
     * @brief Molecular dynamics engine
     *
     * @details This engine is used to perform molecular dynamics simulations.
     */
    class MDEngine : public Engine
    {
       protected:
        resetKinetics::ResetKinetics     _resetKinetics;
        configurator::HybridConfigurator _configurator{};

        std::unique_ptr<integrator::Integrator> _integrator;
        std::unique_ptr<thermostat::Thermostat> _thermostat;
        std::unique_ptr<manostat::Manostat>     _manostat;

       public:
        MDEngine();
        ~MDEngine() override = default;

        void         run() override;
        void         writeOutput() override;
        virtual void takeStep();

        void takeStepBeforeForces();
        void takeStepAfterForces();

        void         calculateForcesWrapper();
        virtual void calculateForces() = 0;

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] resetKinetics::ResetKinetics &getResetKinetics();
        [[nodiscard]] integrator::Integrator       &getIntegrator();
        [[nodiscard]] thermostat::Thermostat       &getThermostat();
        [[nodiscard]] manostat::Manostat           &getManostat();
        [[nodiscard]] output::EnergyOutput         &getInstantEnergyOutput();
        [[nodiscard]] output::MomentumOutput       &getMomentumOutput();
        [[nodiscard]] output::TrajectoryOutput     &getXyzHybridCenterOutput();
        [[nodiscard]] output::TrajectoryOutput     &getVelOutput();
        [[nodiscard]] output::TrajectoryOutput     &getChargeOutput();
        [[nodiscard]] output::VirialOutput         &getVirialOutput();
        [[nodiscard]] output::StressOutput         &getStressOutput();
        [[nodiscard]] output::BoxFileOutput        &getBoxFileOutput();

        [[nodiscard]]
        output::RingPolymerRestartFileOutput &getRingPolymerRstFileOutput();
        [[nodiscard]]
        output::RingPolymerTrajectoryOutput &getRingPolymerXyzOutput();
        [[nodiscard]]
        output::RingPolymerTrajectoryOutput &getRingPolymerVelOutput();
        [[nodiscard]]
        output::RingPolymerTrajectoryOutput &getRingPolymerForceOutput();
        [[nodiscard]]
        output::RingPolymerTrajectoryOutput &getRingPolymerChargeOutput();
        [[nodiscard]]
        output::RingPolymerEnergyOutput &getRingPolymerEnergyOutput();

        /***************************
         * make unique_ptr methods *
         ***************************/

        template <typename T>
        void makeIntegrator(T integrator);
        template <typename T>
        void makeThermostat(T thermostat);
        template <typename T>
        void makeManostat(T manostat);
    };
}   // namespace engine

#ifndef _MD_ENGINE_TPP_
#include "mdEngine.tpp.hpp"   // IWYU pragma: keep - DO NOT MOVE THIS LINE
#endif

#endif   // _MD_ENGINE_HPP_
