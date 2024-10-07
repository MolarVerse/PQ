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
#include "molecularVirial.hpp"
#include "physicalData.hpp"
#include "potential.hpp"
#include "potentialBruteForce.hpp"
#include "potentialCellList.hpp"
#include "simulationBox.hpp"
#include "typeAliases.hpp"
#include "virial.hpp"

#ifdef WITH_KOKKOS
#include "coulombWolf_kokkos.hpp"
#include "lennardJones_kokkos.hpp"
#include "potential_kokkos.hpp"
#include "simulationBox_kokkos.hpp"
#endif

namespace engine
{

    /**
     * @class Engine
     *
     * @brief Contains all the information needed to run the simulation
     *
     */
    class Engine
    {
       protected:
        size_t _step   = 1;
        size_t _nSteps = 0;

        EngineOutput _engineOutput;

        timings::GlobalTimer _timer;

        pq::PhysicalData _averagePhysicalData;

        // clang-format off
        pq::SharedVirial       _virial         = std::make_shared<pq::MolecularVirial>();
        pq::SharedPotential    _potential      = std::make_shared<pq::BruteForcePot>();
        pq::SharedPhysicalData _physicalData   = std::make_shared<pq::PhysicalData>();
        pq::SharedSimBox       _simulationBox  = std::make_shared<pq::SimBox>();
        pq::SharedCellList     _cellList       = std::make_shared<pq::CellList>();
        pq::SharedIntraNonBond _intraNonBonded = std::make_shared<pq::IntraNonBond>();
        pq::SharedForceField   _forceField     = std::make_shared<pq::ForceField>();
        pq::SharedConstraints  _constraints    = std::make_shared<pq::Constraints>();
        // clang-format on

#ifdef WITH_KOKKOS
        pq::KokkosSimBox    _kokkosSimulationBox;
        pq::KokkosLJ        _kokkosLennardJones;
        pq::KokkosWolf      _kokkosCoulombWolf;
        pq::KokkosPotential _kokkosPotential;
#endif

       public:
        Engine()          = default;
        virtual ~Engine() = default;

        virtual void run()         = 0;
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

        /*************************
         * output getter methods *
         *************************/

        [[nodiscard]] EngineOutput      &getEngineOutput();
        [[nodiscard]] pq::LogOutput     &getLogOutput();
        [[nodiscard]] pq::StdoutOutput  &getStdoutOutput();
        [[nodiscard]] pq::TimingsOutput &getTimingsOutput();

        [[nodiscard]] pq::TrajectoryOutput &getXyzOutput();
        [[nodiscard]] pq::TrajectoryOutput &getForceOutput();
        [[nodiscard]] pq::InfoOutput       &getInfoOutput();
        [[nodiscard]] pq::EnergyOutput     &getEnergyOutput();
        [[nodiscard]] pq::RstFileOutput    &getRstFileOutput();

        /***********************
         * get pointer methods *
         ***********************/

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
        [[nodiscard]] pq::KokkosSimBox    &getKokkosSimulationBox();
        [[nodiscard]] pq::KokkosLJ        &getKokkosLennardJones();
        [[nodiscard]] pq::KokkosWolf      &getKokkosCoulombWolf();
        [[nodiscard]] pq::KokkosPotential &getKokkosPotential();
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