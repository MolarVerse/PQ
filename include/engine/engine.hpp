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
#include "typeAliases.hpp"
#include "virial.hpp"

#ifdef WITH_KOKKOS
#include "coulombWolf_kokkos.hpp"
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

#ifdef WITH_KOKKOS
    using KokkosSimulationBox = simulationBox::KokkosSimulationBox;
    using KokkosLennardJones  = potential::KokkosLennardJones;
    using KokkosCoulombWolf   = potential::KokkosCoulombWolf;
    using KokkosPotential     = potential::KokkosPotential;
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

        physicalData::PhysicalData _averagePhysicalData;

        // clang-format off
        pq::SharedVirial       _virial         = std::make_shared<pq::VirialMolecular>();
        pq::SharedPotential    _potential      = std::make_shared<pq::BruteForcePot>();
        pq::SharedPhysicalData _physicalData   = std::make_shared<pq::PhysicalData>();
        pq::SharedSimBox       _simulationBox  = std::make_shared<pq::SimBox>();
        pq::SharedCellList     _cellList       = std::make_shared<pq::CellList>();
        pq::SharedIntraNonBond _intraNonBonded = std::make_shared<pq::IntraNonBond>();
        pq::SharedForceField   _forceField     = std::make_shared<pq::ForceField>();
        pq::SharedConstraints  _constraints    = std::make_shared<pq::Constraints>();
        // clang-format on

#ifdef WITH_KOKKOS
        simulationBox::KokkosSimulationBox _kokkosSimulationBox;
        potential::KokkosLennardJones      _kokkosLennardJones;
        potential::KokkosCoulombWolf       _kokkosCoulombWolf;
        potential::KokkosPotential         _kokkosPotential;
#endif

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

        [[nodiscard]] pq::CellList     &getCellList();
        [[nodiscard]] pq::SimBox       &getSimulationBox();
        [[nodiscard]] pq::PhysicalData &getPhysicalData();
        [[nodiscard]] pq::PhysicalData &getAveragePhysicalData();
        [[nodiscard]] pq::Constraints  &getConstraints();
        [[nodiscard]] pq::ForceField   &getForceField();
        [[nodiscard]] pq::IntraNonBond &getIntraNonBonded();
        [[nodiscard]] pq::Virial       &getVirial();
        [[nodiscard]] pq::Potential    &getPotential();

        [[nodiscard]] EngineOutput          &getEngineOutput();
        [[nodiscard]] output::LogOutput     &getLogOutput();
        [[nodiscard]] output::StdoutOutput  &getStdoutOutput();
        [[nodiscard]] output::TimingsOutput &getTimingsOutput();

        [[nodiscard]] output::TrajectoryOutput &getXyzOutput();
        [[nodiscard]] output::TrajectoryOutput &getForceOutput();
        [[nodiscard]] output::InfoOutput       &getInfoOutput();
        [[nodiscard]] output::EnergyOutput     &getEnergyOutput();
        [[nodiscard]] output::RstFileOutput    &getRstFileOutput();

        [[nodiscard]] pq::ForceField   *getForceFieldPtr();
        [[nodiscard]] pq::Potential    *getPotentialPtr();
        [[nodiscard]] pq::Virial       *getVirialPtr();
        [[nodiscard]] pq::CellList     *getCellListPtr();
        [[nodiscard]] pq::SimBox       *getSimulationBoxPtr();
        [[nodiscard]] pq::PhysicalData *getPhysicalDataPtr();
        [[nodiscard]] pq::Constraints  *getConstraintsPtr();
        [[nodiscard]] pq::IntraNonBond *getIntraNonBondedPtr();

        /******************************
         * get shared pointer methods *
         ******************************/

        [[nodiscard]] pq::SharedForceField   getSharedForceField() const;
        [[nodiscard]] pq::SharedSimBox       getSharedSimulationBox() const;
        [[nodiscard]] pq::SharedPhysicalData getSharedPhysicalData() const;
        [[nodiscard]] pq::SharedCellList     getSharedCellList() const;
        [[nodiscard]] pq::SharedConstraints  getSharedConstraints() const;
        [[nodiscard]] pq::SharedIntraNonBond getSharedIntraNonBonded() const;
        [[nodiscard]] pq::SharedVirial       getSharedVirial() const;
        [[nodiscard]] pq::SharedPotential    getSharedPotential() const;

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