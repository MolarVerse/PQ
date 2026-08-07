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

#include "engine.hpp"
#include "integrator.hpp"
#include "manostat.hpp"
#include "resetKinetics.hpp"
#include "thermostat.hpp"
#include "typeAliases.hpp"
#include "velocityVerlet.hpp"   // for VelocityVerlet

#ifdef WITH_KOKKOS
#include "integrator_kokkos.hpp"
#endif

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
        pq::UniqueIntegrator _integrator = std::make_unique<pq::VelocityVerlet>();
        pq::UniqueThermostat _thermostat = std::make_unique<pq::Thermostat>();
        pq::UniqueManostat   _manostat   = std::make_unique<pq::Manostat>();
        // clang-format off

       public:
        MDEngine()           = default;
        ~MDEngine() override = default;

        void run() override;
        void writeOutput() override;
        virtual void takeStep();

        void takeStepBeforeForces();
        void takeStepAfterForces();

        virtual void calculateForces() = 0;

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] pq::ResetKinetics     &getResetKinetics();
        [[nodiscard]] pq::Integrator        &getIntegrator();
        [[nodiscard]] pq::Thermostat        &getThermostat();
        [[nodiscard]] pq::Manostat          &getManostat();
        [[nodiscard]] pq::EnergyOutput      &getInstantEnergyOutput();
        [[nodiscard]] pq::MomentumOutput    &getMomentumOutput();
        [[nodiscard]] pq::TrajectoryOutput  &getVelOutput();
        [[nodiscard]] pq::TrajectoryOutput  &getChargeOutput();
        [[nodiscard]] pq::VirialOutput      &getVirialOutput();
        [[nodiscard]] pq::StressOutput      &getStressOutput();
        [[nodiscard]] pq::BoxFileOutput     &getBoxFileOutput();
        [[nodiscard]] pq::RPMDRstFileOutput &getRingPolymerRstFileOutput();
        [[nodiscard]] pq::RPMDTrajOutput    &getRingPolymerXyzOutput();
        [[nodiscard]] pq::RPMDTrajOutput    &getRingPolymerVelOutput();
        [[nodiscard]] pq::RPMDTrajOutput    &getRingPolymerForceOutput();
        [[nodiscard]] pq::RPMDTrajOutput    &getRingPolymerChargeOutput();
        [[nodiscard]] pq::RPMDEnergyOutput  &getRingPolymerEnergyOutput();

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

#include "mdEngine.tpp.hpp"   // DO NOT MOVE THIS LINE

#endif   // _MD_ENGINE_HPP_