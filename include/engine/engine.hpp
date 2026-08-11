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
#include "interWater.hpp"
#include "intraNonBonded.hpp"
#include "intraWater.hpp"
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

        physicalData::PhysicalData _averagePhysicalData;

        std::shared_ptr<virial::Virial>                 _virial;
        std::shared_ptr<potential::Potential>           _potential;
        std::shared_ptr<physicalData::PhysicalData>     _physicalData;
        std::shared_ptr<simulationBox::SimulationBox>   _simulationBox;
        std::shared_ptr<simulationBox::CellList>        _cellList;
        std::shared_ptr<intraNonBonded::IntraNonBonded> _intraNonBonded;
        std::shared_ptr<forceField::ForceField>         _forceField;
        std::shared_ptr<constraints::Constraints>       _constraints;

        std::unique_ptr<waterModel::IntraWater> _intraWater =
            std::make_unique<waterModel::IntraWater>();
        std::unique_ptr<waterModel::InterWater> _interWater =
            std::make_unique<waterModel::InterWater>();

#ifdef WITH_KOKKOS
        simulationBox::KokkosSimulationBox _kokkosSimulationBox;
        potential::KokkosLennardJones      _kokkosLennardJones;
        potential::KokkosCoulombWolf       _kokkosCoulombWolf;
        potential::KokkosPotential         _kokkosPotential;
#endif

       public:
        Engine();
        virtual ~Engine() = default;

        virtual void run()         = 0;
        virtual void writeOutput() = 0;
        void         deleteTmpFiles();

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

        [[nodiscard]]
        const std::shared_ptr<simulationBox::CellList> &getCellList() const;

        [[nodiscard]] simulationBox::SimulationBox   &getSimulationBox();
        [[nodiscard]] physicalData::PhysicalData     &getPhysicalData();
        [[nodiscard]] physicalData::PhysicalData     &getAveragePhysicalData();
        [[nodiscard]] constraints::Constraints       &getConstraints();
        [[nodiscard]] forceField::ForceField         &getForceField();
        [[nodiscard]] intraNonBonded::IntraNonBonded &getIntraNonBonded();
        [[nodiscard]] virial::Virial                 &getVirial();
        [[nodiscard]] potential::Potential           &getPotential();

        /*************************
         * output getter methods *
         *************************/

        [[nodiscard]] EngineOutput          &getEngineOutput();
        [[nodiscard]] output::LogOutput     &getLogOutput();
        [[nodiscard]] output::StdoutOutput  &getStdoutOutput();
        [[nodiscard]] output::TimingsOutput &getTimingsOutput();

        [[nodiscard]] output::TrajectoryOutput &getXyzOutput();
        [[nodiscard]] output::TrajectoryOutput &getForceOutput();
        [[nodiscard]] output::InfoOutput       &getInfoOutput();
        [[nodiscard]] output::EnergyOutput     &getEnergyOutput();
        [[nodiscard]] output::RstFileOutput    &getRstFileOutput();

        /***********************
         * get pointer methods *
         ***********************/

        [[nodiscard]] forceField::ForceField         *getForceFieldPtr();
        [[nodiscard]] potential::Potential           *getPotentialPtr();
        [[nodiscard]] virial::Virial                 *getVirialPtr();
        [[nodiscard]] simulationBox::SimulationBox   *getSimulationBoxPtr();
        [[nodiscard]] physicalData::PhysicalData     *getPhysicalDataPtr();
        [[nodiscard]] constraints::Constraints       *getConstraintsPtr();
        [[nodiscard]] intraNonBonded::IntraNonBonded *getIntraNonBondedPtr();

        /******************************
         * get shared pointer methods *
         ******************************/

        [[nodiscard]]
        std::shared_ptr<forceField::ForceField> getSharedForceField() const;
        [[nodiscard]] std::shared_ptr<simulationBox::SimulationBox> getSharedSimulationBox(
        ) const;
        [[nodiscard]]
        std::shared_ptr<physicalData::PhysicalData> getSharedPhysicalData(
        ) const;
        [[nodiscard]] std::shared_ptr<simulationBox::CellList> getSharedCellList(
        ) const;
        [[nodiscard]]
        std::shared_ptr<constraints::Constraints> getSharedConstraints() const;
        [[nodiscard]]
        std::shared_ptr<intraNonBonded::IntraNonBonded> getSharedIntraNonBonded(
        ) const;
        [[nodiscard]] std::shared_ptr<virial::Virial> getSharedVirial() const;
        [[nodiscard]] std::shared_ptr<potential::Potential> getSharedPotential(
        ) const;

        /***************************
         * make unique_ptr methods *
         ***************************/

        template <typename T>
        void makePotential(T);
        template <typename T>
        void makeVirial(T virial);
        template <typename T>
        void makeIntraWater(T &&);

        /********************************
         * standard getters and setters *
         ********************************/

        [[nodiscard]] size_t                getStep() const { return _step; }
        [[nodiscard]] timings::GlobalTimer &getTimer() { return _timer; }

        void setTimer(const timings::GlobalTimer &timer) { _timer = timer; }
        void setInterWater(std::unique_ptr<waterModel::InterWater> interWater);

#ifdef WITH_KOKKOS
        [[nodiscard]] simulationBox::KokkosSimulationBox &getKokkosSimulationBox(
        );
        [[nodiscard]] potential::KokkosLennardJones &getKokkosLennardJones();
        [[nodiscard]] potential::KokkosCoulombWolf  &getKokkosCoulombWolf();
        [[nodiscard]] potential::KokkosPotential    &getKokkosPotential();
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

#ifndef _ENGINE_TPP_
#include "engine.tpp.hpp"   // IWYU pragma: keep - DO NOT MOVE THIS LINE!
#endif

#endif   // _ENGINE_HPP_
