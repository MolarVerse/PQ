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
#include "typeAliases.hpp"
#include "velocityVerlet.hpp"
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
        pq::ResetKinetics _resetKinetics;

        // clang-format off
        pq::UniqueIntegrator _integrator = std::make_unique<integrator::VelocityVerlet>();
        pq::UniqueThermostat _thermostat = std::make_unique<thermostat::Thermostat>();
        pq::UniqueManostat   _manostat   = std::make_unique<manostat::Manostat>();
        // clang-format on

       public:
        MDEngine()           = default;
        ~MDEngine() override = default;

        void         run() override;
        void         writeOutput() override;
        virtual void takeStep();

        void takeStepBeforeForces();
        void takeStepAfterForces();

        virtual void calculateForces() = 0;

        /***************************
         * standard getter methods *
         ***************************/

        // clang-format off
        [[nodiscard]] resetKinetics::ResetKinetics &getResetKinetics();
        [[nodiscard]] integrator::Integrator       &getIntegrator();
        [[nodiscard]] thermostat::Thermostat       &getThermostat();
        [[nodiscard]] manostat::Manostat           &getManostat();
        [[nodiscard]] output::EnergyOutput         &getInstantEnergyOutput();
        [[nodiscard]] output::MomentumOutput       &getMomentumOutput();
        [[nodiscard]] output::TrajectoryOutput     &getVelOutput();
        [[nodiscard]] output::TrajectoryOutput     &getChargeOutput();
        [[nodiscard]] output::VirialOutput         &getVirialOutput();
        [[nodiscard]] output::StressOutput         &getStressOutput();
        [[nodiscard]] output::BoxFileOutput        &getBoxFileOutput();
        [[nodiscard]] output::RingPolymerRestartFileOutput &getRingPolymerRstFileOutput();
        [[nodiscard]] output::RingPolymerTrajectoryOutput    &getRingPolymerXyzOutput();
        [[nodiscard]] output::RingPolymerTrajectoryOutput    &getRingPolymerVelOutput();
        [[nodiscard]] output::RingPolymerTrajectoryOutput    &getRingPolymerForceOutput();
        [[nodiscard]] output::RingPolymerTrajectoryOutput    &getRingPolymerChargeOutput();
        [[nodiscard]] output::RingPolymerEnergyOutput  &getRingPolymerEnergyOutput();
        // clang-format on

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