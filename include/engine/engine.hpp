#ifndef _ENGINE_HPP_

#define _ENGINE_HPP_

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

#include <cstddef>   // for size_t
#include <memory>

namespace output
{
    class EnergyOutput;
    class InfoOutput;
    class LogOutput;
    class RstFileOutput;
    class StdoutOutput;
    class TrajectoryOutput;
}   // namespace output

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
        size_t _step = 1;

        EngineOutput                   _engineOutput;
        timings::Timings               _timings;
        simulationBox::CellList        _cellList;
        simulationBox::SimulationBox   _simulationBox;
        physicalData::PhysicalData     _physicalData;
        physicalData::PhysicalData     _averagePhysicalData;
        constraints::Constraints       _constraints;
        forceField::ForceField         _forceField;
        intraNonBonded::IntraNonBonded _intraNonBonded;
        resetKinetics::ResetKinetics   _resetKinetics;

        std::unique_ptr<integrator::Integrator> _integrator = std::make_unique<integrator::VelocityVerlet>();
        std::unique_ptr<thermostat::Thermostat> _thermostat = std::make_unique<thermostat::Thermostat>();
        std::unique_ptr<manostat::Manostat>     _manostat   = std::make_unique<manostat::Manostat>();
        std::unique_ptr<virial::Virial>         _virial     = std::make_unique<virial::VirialMolecular>();
        std::unique_ptr<potential::Potential>   _potential  = std::make_unique<potential::PotentialBruteForce>();

      public:
        Engine()          = default;
        virtual ~Engine() = default;

        virtual void run();
        virtual void takeStep(){};
        virtual void writeOutput();

        [[nodiscard]] bool isForceFieldNonCoulombicsActivated() const { return _forceField.isNonCoulombicActivated(); }
        [[nodiscard]] bool isGuffActivated() const { return !_forceField.isNonCoulombicActivated(); }
        [[nodiscard]] bool isCellListActivated() const { return _cellList.isActive(); }
        [[nodiscard]] bool isConstraintsActivated() const { return _constraints.isActive(); }
        [[nodiscard]] bool isIntraNonBondedActivated() const { return _intraNonBonded.isActive(); }

        /************************************
         *                                  *
         * standard make unique_ptr methods *
         *                                  *
         ************************************/

        template <typename T>
        void makeIntegrator(T integrator)
        {
            _integrator = std::make_unique<T>(integrator);
        }
        template <typename T>
        void makePotential(T)
        {
            _potential = std::make_unique<T>();
        }
        template <typename T>
        void makeThermostat(T thermostat)
        {
            _thermostat = std::make_unique<T>(thermostat);
        }
        template <typename T>
        void makeManostat(T manostat)
        {
            _manostat = std::make_unique<T>(manostat);
        }
        template <typename T>
        void makeVirial(T virial)
        {
            _virial = std::make_unique<T>(virial);
        }

        /***************************
         *                         *
         * standard getter methods *
         *                         *
         ***************************/

        [[nodiscard]] timings::Timings               &getTimings() { return _timings; }
        [[nodiscard]] simulationBox::CellList        &getCellList() { return _cellList; }
        [[nodiscard]] simulationBox::SimulationBox   &getSimulationBox() { return _simulationBox; }
        [[nodiscard]] physicalData::PhysicalData     &getPhysicalData() { return _physicalData; }
        [[nodiscard]] physicalData::PhysicalData     &getAveragePhysicalData() { return _averagePhysicalData; }
        [[nodiscard]] constraints::Constraints       &getConstraints() { return _constraints; }
        [[nodiscard]] forceField::ForceField         &getForceField() { return _forceField; }
        [[nodiscard]] intraNonBonded::IntraNonBonded &getIntraNonBonded() { return _intraNonBonded; }
        [[nodiscard]] resetKinetics::ResetKinetics   &getResetKinetics() { return _resetKinetics; }

        [[nodiscard]] virial::Virial         &getVirial() { return *_virial; }
        [[nodiscard]] integrator::Integrator &getIntegrator() { return *_integrator; }
        [[nodiscard]] potential::Potential   &getPotential() { return *_potential; }
        [[nodiscard]] thermostat::Thermostat &getThermostat() { return *_thermostat; }
        [[nodiscard]] manostat::Manostat     &getManostat() { return *_manostat; }

        [[nodiscard]] EngineOutput                         &getEngineOutput() { return _engineOutput; }
        [[nodiscard]] output::EnergyOutput                 &getEnergyOutput() { return _engineOutput.getEnergyOutput(); }
        [[nodiscard]] output::MomentumOutput               &getMomentumOutput() { return _engineOutput.getMomentumOutput(); }
        [[nodiscard]] output::TrajectoryOutput             &getXyzOutput() { return _engineOutput.getXyzOutput(); }
        [[nodiscard]] output::TrajectoryOutput             &getVelOutput() { return _engineOutput.getVelOutput(); }
        [[nodiscard]] output::TrajectoryOutput             &getForceOutput() { return _engineOutput.getForceOutput(); }
        [[nodiscard]] output::TrajectoryOutput             &getChargeOutput() { return _engineOutput.getChargeOutput(); }
        [[nodiscard]] output::LogOutput                    &getLogOutput() { return _engineOutput.getLogOutput(); }
        [[nodiscard]] output::StdoutOutput                 &getStdoutOutput() { return _engineOutput.getStdoutOutput(); }
        [[nodiscard]] output::RstFileOutput                &getRstFileOutput() { return _engineOutput.getRstFileOutput(); }
        [[nodiscard]] output::InfoOutput                   &getInfoOutput() { return _engineOutput.getInfoOutput(); }
        [[nodiscard]] output::RingPolymerRestartFileOutput &getRingPolymerRstFileOutput()
        {
            return _engineOutput.getRingPolymerRstFileOutput();
        }
        [[nodiscard]] output::RingPolymerTrajectoryOutput &getRingPolymerXyzOutput()
        {
            return _engineOutput.getRingPolymerXyzOutput();
        }
        [[nodiscard]] output::RingPolymerTrajectoryOutput &getRingPolymerVelOutput()
        {
            return _engineOutput.getRingPolymerVelOutput();
        }
        [[nodiscard]] output::RingPolymerTrajectoryOutput &getRingPolymerForceOutput()
        {
            return _engineOutput.getRingPolymerForceOutput();
        }
        [[nodiscard]] output::RingPolymerTrajectoryOutput &getRingPolymerChargeOutput()
        {
            return _engineOutput.getRingPolymerChargeOutput();
        }

        [[nodiscard]] forceField::ForceField *getForceFieldPtr() { return &_forceField; }
    };

}   // namespace engine

#endif   // _ENGINE_HPP_