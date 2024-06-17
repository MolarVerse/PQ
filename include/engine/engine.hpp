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
#include "globalTimer.hpp"
#include "intraNonBonded.hpp"
#include "physicalData.hpp"
#include "potential.hpp"
#include "simulationBox.hpp"
#include "virial.hpp"

#ifdef WITH_KOKKOS
#include "coulombWolf_kokkos.hpp"
#include "lennardJones_kokkos.hpp"
#include "potential_kokkos.hpp"
#include "simulationBox_kokkos.hpp"
#endif

#ifdef WITH_CUDA
#include "potential.cuh"
#include "simulationBox_cuda.cuh"
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
    class TimingsOutput;                  // forward declaration

}   // namespace output

namespace engine
{
    using RPMDRestartFileOutput = output::RingPolymerRestartFileOutput;
    using RPMDTrajectoryOutput  = output::RingPolymerTrajectoryOutput;
    using RPMDVelOutput         = output::RingPolymerTrajectoryOutput;
    using RPMDForceOutput       = output::RingPolymerTrajectoryOutput;
    using RPMDChargeOutput      = output::RingPolymerTrajectoryOutput;
    using RPMDEnergyOutput      = output::RingPolymerEnergyOutput;

    using UniqueVirial    = std::unique_ptr<virial::Virial>;
    using UniquePotential = std::unique_ptr<potential::Potential>;
    using BruteForce      = potential::PotentialBruteForce;

#ifdef WITH_KOKKOS
    using KokkosSimulationBox = simulationBox::KokkosSimulationBox;
    using KokkosLennardJones  = potential::KokkosLennardJones;
    using KokkosCoulombWolf   = potential::KokkosCoulombWolf;
    using KokkosPotential     = potential::KokkosPotential;
#endif

#ifdef WITH_CUDA
    using PotentialCuda     = potential::PotentialCuda;
    using SimulationBoxCuda = simulationBox::SimulationBoxCuda;
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

        timings::GlobalTimer _timer;

        simulationBox::CellList        _cellList;
        simulationBox::SimulationBox   _simulationBox;
        physicalData::PhysicalData     _physicalData;
        physicalData::PhysicalData     _averagePhysicalData;
        constraints::Constraints       _constraints;
        forceField::ForceField         _forceField;
        intraNonBonded::IntraNonBonded _intraNonBonded;

#ifdef WITH_KOKKOS
        simulationBox::KokkosSimulationBox _kokkosSimulationBox;
        potential::KokkosLennardJones      _kokkosLennardJones;
        potential::KokkosCoulombWolf       _kokkosCoulombWolf;
        potential::KokkosPotential         _kokkosPotential;
#endif

#ifdef WITH_CUDA
        potential::PotentialCuda         _cudaPotential;
        simulationBox::SimulationBoxCuda _cudaSimulationBox;
#endif

        UniqueVirial    _virial = std::make_unique<virial::VirialMolecular>();
        UniquePotential _potential = std::make_unique<BruteForce>();

       public:
        Engine()          = default;
        virtual ~Engine() = default;

        virtual void run()         = 0;
        virtual void takeStep()    = 0;
        virtual void writeOutput() = 0;

        void addTimer(const timings::Timer &timings);

        [[nodiscard]] double calculateTotalSimulationTime() const;

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
        [[nodiscard]] virial::Virial                 &getVirial();
        [[nodiscard]] potential::Potential           &getPotential();

        [[nodiscard]] EngineOutput          &getEngineOutput();
        [[nodiscard]] output::LogOutput     &getLogOutput();
        [[nodiscard]] output::StdoutOutput  &getStdoutOutput();
        [[nodiscard]] output::TimingsOutput &getTimingsOutput();

        [[nodiscard]] forceField::ForceField         *getForceFieldPtr();
        [[nodiscard]] potential::Potential           *getPotentialPtr();
        [[nodiscard]] virial::Virial                 *getVirialPtr();
        [[nodiscard]] simulationBox::CellList        *getCellListPtr();
        [[nodiscard]] simulationBox::SimulationBox   *getSimulationBoxPtr();
        [[nodiscard]] physicalData::PhysicalData     *getPhysicalDataPtr();
        [[nodiscard]] constraints::Constraints       *getConstraintsPtr();
        [[nodiscard]] intraNonBonded::IntraNonBonded *getIntraNonBondedPtr();

        /***************************
         * make unique_ptr methods *
         ***************************/

        template <typename T>
        void makePotential(T);
        template <typename T>
        void makeVirial(T virial);

        /********************************
         * standard getters and setters *
         ********************************/

        [[nodiscard]] size_t                getStep() const { return _step; }
        [[nodiscard]] timings::GlobalTimer &getTimer() { return _timer; }

        void setTimer(const timings::GlobalTimer &timer) { _timer = timer; }

#ifdef WITH_KOKKOS
        [[nodiscard]] KokkosSimulationBox &getKokkosSimulationBox();
        [[nodiscard]] KokkosLennardJones  &getKokkosLennardJones();
        [[nodiscard]] KokkosCoulombWolf   &getKokkosCoulombWolf();
        [[nodiscard]] KokkosPotential     &getKokkosPotential();
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
#endif
    };
}   // namespace engine

#include "engine.tpp.hpp"   // DO NOT MOVE THIS LINE!

#endif   // _ENGINE_HPP_