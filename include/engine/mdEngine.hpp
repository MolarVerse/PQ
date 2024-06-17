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

namespace engine
{
    using VelocityVerlet   = integrator::VelocityVerlet;
    using UniqueIntegrator = std::unique_ptr<integrator::Integrator>;

    using Thermostat       = thermostat::Thermostat;
    using UniqueThermostat = std::unique_ptr<thermostat::Thermostat>;

    using UniqueManostat = std::unique_ptr<manostat::Manostat>;

    /**
     * @brief Molecular dynamics engine
     *
     * @details This engine is used to perform molecular dynamics simulations.
     */
    class MDEngine : public Engine
    {
       protected:
        resetKinetics::ResetKinetics _resetKinetics;

        UniqueIntegrator _integrator = std::make_unique<VelocityVerlet>();
        UniqueThermostat _thermostat = std::make_unique<Thermostat>();
        UniqueManostat   _manostat   = std::make_unique<manostat::Manostat>();

       public:
        MDEngine()           = default;
        ~MDEngine() override = default;

        void run() override;
        void writeOutput() override;

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] resetKinetics::ResetKinetics &getResetKinetics();
        [[nodiscard]] integrator::Integrator       &getIntegrator();
        [[nodiscard]] thermostat::Thermostat       &getThermostat();
        [[nodiscard]] manostat::Manostat           &getManostat();
        [[nodiscard]] output::EnergyOutput         &getEnergyOutput();
        [[nodiscard]] output::EnergyOutput         &getInstantEnergyOutput();
        [[nodiscard]] output::MomentumOutput       &getMomentumOutput();
        [[nodiscard]] output::TrajectoryOutput     &getXyzOutput();
        [[nodiscard]] output::TrajectoryOutput     &getVelOutput();
        [[nodiscard]] output::TrajectoryOutput     &getForceOutput();
        [[nodiscard]] output::TrajectoryOutput     &getChargeOutput();
        [[nodiscard]] output::RstFileOutput        &getRstFileOutput();
        [[nodiscard]] output::InfoOutput           &getInfoOutput();
        [[nodiscard]] output::VirialOutput         &getVirialOutput();
        [[nodiscard]] output::StressOutput         &getStressOutput();
        [[nodiscard]] output::BoxFileOutput        &getBoxFileOutput();
        [[nodiscard]] RPMDRestartFileOutput &getRingPolymerRstFileOutput();
        [[nodiscard]] RPMDTrajectoryOutput  &getRingPolymerXyzOutput();
        [[nodiscard]] RPMDTrajectoryOutput  &getRingPolymerVelOutput();
        [[nodiscard]] RPMDTrajectoryOutput  &getRingPolymerForceOutput();
        [[nodiscard]] RPMDTrajectoryOutput  &getRingPolymerChargeOutput();
        [[nodiscard]] RPMDEnergyOutput      &getRingPolymerEnergyOutput();

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