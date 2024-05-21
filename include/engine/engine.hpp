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

#ifndef _ENGINE_HPP_

#define _ENGINE_HPP_

#include <cstddef>   // for size_t
#include <memory>

#include "celllist.hpp"
#include "constraints.hpp"
#include "engineOutput.hpp"
#include "forceFieldClass.hpp"
#include "integrator.hpp"
#include "intraNonBonded.hpp"
#include "manostat.hpp"
#include "physicalData.hpp"
#include "potential.hpp"
#include "resetKinetics.hpp"
#include "simulationBox.hpp"
#include "thermostat.hpp"
#include "timings.hpp"
#include "virial.hpp"

#ifdef WITH_KOKKOS
#include "coulombWolf_kokkos.hpp"
#include "integrator_kokkos.hpp"
#include "lennardJones_kokkos.hpp"
#include "potential_kokkos.hpp"
#include "simulationBox_kokkos.hpp"
#endif

namespace output
{
    class EnergyOutput;                   // forward declaration
    class InfoOutput;                     // forward declaration
    class LogOutput;                      // forward declaration
    class RstFileOutput;                  // forward declaration
    class StdoutOutput;                   // forward declaration
    class TrajectoryOutput;               // forward declaration
    class MomentumOutput;                 // forward declaration
    class VirialOutput;                   // forward declaration
    class StressOutput;                   // forward declaration
    class BoxFileOutput;                  // forward declaration
    class RingPolymerRestartFileOutput;   // forward declaration
    class RingPolymerTrajectoryOutput;    // forward declaration

}   // namespace output

namespace engine
{
    using RPRestartFileOutput = output::RingPolymerRestartFileOutput;
    using RPTrajectoryOutput  = output::RingPolymerTrajectoryOutput;
    using RPVelOutput         = output::RingPolymerTrajectoryOutput;
    using RPForceOutput       = output::RingPolymerTrajectoryOutput;
    using RPChargeOutput      = output::RingPolymerTrajectoryOutput;
    using RPEnergyOutput      = output::RingPolymerEnergyOutput;

    using UniqueIntegrator = std::unique_ptr<integrator::Integrator>;
    using VelocityVerlet   = integrator::VelocityVerlet;

    using UniqueThermostat = std::unique_ptr<thermostat::Thermostat>;
    using Thermostat       = thermostat::Thermostat;

    using UniqueManostat = std::unique_ptr<manostat::Manostat>;
    using UniqueVirial   = std::unique_ptr<virial::Virial>;

    using UniquePotential = std::unique_ptr<potential::Potential>;
    using BruteForce      = potential::PotentialBruteForce;

#ifdef WITH_KOKKOS
    using KokkosSimulationBox  = simulationBox::KokkosSimulationBox;
    using KokkosLennardJones   = potential::KokkosLennardJones;
    using KokkosCoulombWolf    = potential::KokkosCoulombWolf;
    using KokkosPotential      = potential::KokkosPotential;
    using KokkosVelocityVerlet = integrator::KokkosVelocityVerlet;
#endif

    /**
     * @class Engine
     *
     * @brief Contains all the information needed to run the simulation
     *
     */
    class Engine
    {
       protected:
        size_t _step = 1;

        EngineOutput _engineOutput;

        timings::Timings              _timings;
        std::vector<timings::Timings> _timingsSections;

        simulationBox::CellList        _cellList;
        simulationBox::SimulationBox   _simulationBox;
        physicalData::PhysicalData     _physicalData;
        physicalData::PhysicalData     _averagePhysicalData;
        constraints::Constraints       _constraints;
        forceField::ForceField         _forceField;
        intraNonBonded::IntraNonBonded _intraNonBonded;
        resetKinetics::ResetKinetics   _resetKinetics;

#ifdef WITH_KOKKOS
        simulationBox::KokkosSimulationBox _kokkosSimulationBox;
        potential::KokkosLennardJones      _kokkosLennardJones;
        potential::KokkosCoulombWolf       _kokkosCoulombWolf;
        potential::KokkosPotential         _kokkosPotential;
        integrator::KokkosVelocityVerlet   _kokkosVelocityVerlet;
#endif

        UniqueIntegrator _integrator = std::make_unique<VelocityVerlet>();
        UniqueThermostat _thermostat = std::make_unique<Thermostat>();
        UniqueManostat   _manostat   = std::make_unique<manostat::Manostat>();
        UniqueVirial     _virial = std::make_unique<virial::VirialMolecular>();
        UniquePotential  _potential = std::make_unique<BruteForce>();

       public:
        Engine()          = default;
        virtual ~Engine() = default;

        virtual void run();
        virtual void writeOutput();

        void addTimingsSection(const timings::Timings &timings);

        // virtual function to be overwritten by derived classes
        virtual void takeStep() {};

        /**********************************
         * information about active parts *
         **********************************/

        [[nodiscard]] bool isForceFieldNonCoulombicsActivated() const;
        [[nodiscard]] bool isGuffActivated() const;
        [[nodiscard]] bool isCellListActivated() const;
        [[nodiscard]] bool isConstraintsActivated() const;
        [[nodiscard]] bool isIntraNonBondedActivated() const;

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] simulationBox::CellList        &getCellList();
        [[nodiscard]] simulationBox::SimulationBox   &getSimulationBox();
        [[nodiscard]] physicalData::PhysicalData     &getPhysicalData();
        [[nodiscard]] physicalData::PhysicalData     &getAveragePhysicalData();
        [[nodiscard]] constraints::Constraints       &getConstraints();
        [[nodiscard]] forceField::ForceField         &getForceField();
        [[nodiscard]] intraNonBonded::IntraNonBonded &getIntraNonBonded();
        [[nodiscard]] resetKinetics::ResetKinetics   &getResetKinetics();
        [[nodiscard]] virial::Virial                 &getVirial();
        [[nodiscard]] integrator::Integrator         &getIntegrator();
        [[nodiscard]] potential::Potential           &getPotential();
        [[nodiscard]] thermostat::Thermostat         &getThermostat();
        [[nodiscard]] manostat::Manostat             &getManostat();
        [[nodiscard]] EngineOutput                   &getEngineOutput();
        [[nodiscard]] output::EnergyOutput           &getEnergyOutput();
        [[nodiscard]] output::EnergyOutput           &getInstantEnergyOutput();
        [[nodiscard]] output::MomentumOutput         &getMomentumOutput();
        [[nodiscard]] output::TrajectoryOutput       &getXyzOutput();
        [[nodiscard]] output::TrajectoryOutput       &getVelOutput();
        [[nodiscard]] output::TrajectoryOutput       &getForceOutput();
        [[nodiscard]] output::TrajectoryOutput       &getChargeOutput();
        [[nodiscard]] output::LogOutput              &getLogOutput();
        [[nodiscard]] output::StdoutOutput           &getStdoutOutput();
        [[nodiscard]] output::RstFileOutput          &getRstFileOutput();
        [[nodiscard]] output::InfoOutput             &getInfoOutput();
        [[nodiscard]] output::VirialOutput           &getVirialOutput();
        [[nodiscard]] output::StressOutput           &getStressOutput();
        [[nodiscard]] output::BoxFileOutput          &getBoxFileOutput();
        [[nodiscard]] RPRestartFileOutput &getRingPolymerRstFileOutput();
        [[nodiscard]] RPTrajectoryOutput  &getRingPolymerXyzOutput();
        [[nodiscard]] RPTrajectoryOutput  &getRingPolymerVelOutput();
        [[nodiscard]] RPTrajectoryOutput  &getRingPolymerForceOutput();
        [[nodiscard]] RPTrajectoryOutput  &getRingPolymerChargeOutput();
        [[nodiscard]] RPEnergyOutput      &getRingPolymerEnergyOutput();

        [[nodiscard]] forceField::ForceField *getForceFieldPtr();

        /***************************
         * make unique_ptr methods *
         ***************************/

        template <typename T>
        void makeIntegrator(T integrator);
        template <typename T>
        void makePotential(T);
        template <typename T>
        void makeThermostat(T thermostat);
        template <typename T>
        void makeManostat(T manostat);
        template <typename T>
        void makeVirial(T virial);

        /********************************
         * standard getters and setters *
         ********************************/

        [[nodiscard]] size_t            getStep() const { return _step; }
        [[nodiscard]] timings::Timings &getTimings() { return _timings; }

        void setTimings(const timings::Timings &timings) { _timings = timings; }

        /***************************
         *                         *
         * standard getter methods *
         *                         *
         ***************************/

#ifdef WITH_KOKKOS
        [[nodiscard]] KokkosSimulationBox  &getKokkosSimulationBox();
        [[nodiscard]] KokkosLennardJones   &getKokkosLennardJones();
        [[nodiscard]] KokkosCoulombWolf    &getKokkosCoulombWolf();
        [[nodiscard]] KokkosPotential      &getKokkosPotential();
        [[nodiscard]] KokkosVelocityVerlet &getKokkosVelocityVerlet();
        void initKokkosSimulationBox(const size_t numAtoms);
        void initKokkosLennardJones(const size_t numAtomTypes);
        void initKokkosCoulombWolf(
            const double coulombRadiusCutOff,
            const double kappa,
            const double wolfParameter1,
            const double wolfParameter2,
            const double wolfParameter3,
            const double prefactor
        );
        void initKokkosPotential();
        void initKokkosVelocityVerlet(
            const double dt,
            const double velocityFactor,
            const double timeFactor
        );
#endif
    };
}   // namespace engine

#include "engine.tpp.hpp"   // DO NOT MOVE THIS LINE!

#endif   // _ENGINE_HPP_