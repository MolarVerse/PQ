#ifndef _ENGINE_HPP_

#define _ENGINE_HPP_

#include "celllist.hpp"
#include "constraints.hpp"
#include "engineOutput.hpp"
#include "forceField.hpp"
#include "integrator.hpp"
#include "manostat.hpp"
#include "physicalData.hpp"
#include "potential.hpp"
#include "resetKinetics.hpp"
#include "settings.hpp"
#include "simulationBox.hpp"
#include "thermostat.hpp"
#include "timings.hpp"
#include "virial.hpp"

#include <memory>

namespace engine
{
    class Engine;
}

/**
 * @class Engine
 *
 * @brief Contains all the information needed to run the simulation
 *
 */
class engine::Engine
{
  private:
    size_t _step = 1;

    settings::Settings           _settings;
    timings::Timings             _timings;
    simulationBox::CellList      _cellList;
    simulationBox::SimulationBox _simulationBox;
    physicalData::PhysicalData   _physicalData;
    physicalData::PhysicalData   _averagePhysicalData;
    constraints::Constraints     _constraints;
    forceField::ForceField       _forceField;
    engine::EngineOutput         _engineOutput;

    std::unique_ptr<integrator::Integrator>       _integrator    = std::make_unique<integrator::VelocityVerlet>();
    std::unique_ptr<potential::Potential>         _potential     = std::make_unique<potential::PotentialBruteForce>();
    std::unique_ptr<thermostat::Thermostat>       _thermostat    = std::make_unique<thermostat::Thermostat>();
    std::unique_ptr<manostat::Manostat>           _manostat      = std::make_unique<manostat::Manostat>();
    std::unique_ptr<virial::Virial>               _virial        = std::make_unique<virial::VirialMolecular>();
    std::unique_ptr<resetKinetics::ResetKinetics> _resetKinetics = std::make_unique<resetKinetics::ResetKinetics>();

  public:
    void run();
    void takeStep();
    void writeOutput();

    template <typename T> void makeIntegrator(T integrator) { _integrator = std::make_unique<T>(integrator); }
    template <typename T> void makePotential(T potential) { _potential = std::make_unique<T>(potential); }
    template <typename T> void makeThermostat(T thermostat) { _thermostat = std::make_unique<T>(thermostat); }
    template <typename T> void makeManostat(T manostat) { _manostat = std::make_unique<T>(manostat); }
    template <typename T> void makeVirial(T virial) { _virial = std::make_unique<T>(virial); }
    template <typename T> void makeResetKinetics(T resetKinetics) { _resetKinetics = std::make_unique<T>(resetKinetics); }

    /***************************
     *                         *
     * standard getter methods *
     *                         *
     ***************************/

    settings::Settings           &getSettings() { return _settings; }
    timings::Timings             &getTimings() { return _timings; }
    simulationBox::CellList      &getCellList() { return _cellList; }
    simulationBox::SimulationBox &getSimulationBox() { return _simulationBox; }
    physicalData::PhysicalData   &getPhysicalData() { return _physicalData; }
    physicalData::PhysicalData   &getAveragePhysicalData() { return _averagePhysicalData; }
    virial::Virial               &getVirial() { return *_virial; }
    integrator::Integrator       &getIntegrator() { return *_integrator; }
    constraints::Constraints     &getConstraints() { return _constraints; }
    forceField::ForceField       &getForceField() { return _forceField; }
    potential::Potential         &getPotential() { return *_potential; }
    thermostat::Thermostat       &getThermostat() { return *_thermostat; }
    manostat::Manostat           &getManostat() { return *_manostat; }
    resetKinetics::ResetKinetics &getResetKinetics() { return *_resetKinetics; }

    engine::EngineOutput     &getEngineOutput() { return _engineOutput; }
    output::EnergyOutput     &getEnergyOutput() { return _engineOutput.getEnergyOutput(); }
    output::TrajectoryOutput &getXyzOutput() { return _engineOutput.getXyzOutput(); }
    output::TrajectoryOutput &getVelOutput() { return _engineOutput.getVelOutput(); }
    output::TrajectoryOutput &getForceOutput() { return _engineOutput.getForceOutput(); }
    output::TrajectoryOutput &getChargeOutput() { return _engineOutput.getChargeOutput(); }
    output::LogOutput        &getLogOutput() { return _engineOutput.getLogOutput(); }
    output::StdoutOutput     &getStdoutOutput() { return _engineOutput.getStdoutOutput(); }
    output::RstFileOutput    &getRstFileOutput() { return _engineOutput.getRstFileOutput(); }
    output::InfoOutput       &getInfoOutput() { return _engineOutput.getInfoOutput(); }

    forceField::ForceField *getForceFieldPtr() { return &_forceField; }
};

#endif   // _ENGINE_HPP_