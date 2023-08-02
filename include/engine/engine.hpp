#ifndef _ENGINE_HPP_

#define _ENGINE_HPP_

#include "celllist.hpp"
#include "constraints.hpp"
#include "energyOutput.hpp"
#include "forceField.hpp"
#include "infoOutput.hpp"
#include "integrator.hpp"
#include "logOutput.hpp"
#include "manostat.hpp"
#include "physicalData.hpp"
#include "potential.hpp"
#include "resetKinetics.hpp"
#include "rstFileOutput.hpp"
#include "settings.hpp"
#include "simulationBox.hpp"
#include "stdoutOutput.hpp"
#include "thermostat.hpp"
#include "timings.hpp"
#include "trajectoryOutput.hpp"
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

    std::unique_ptr<integrator::Integrator>       _integrator    = std::make_unique<integrator::VelocityVerlet>();
    std::unique_ptr<potential::Potential>         _potential     = std::make_unique<potential::PotentialBruteForce>();
    std::unique_ptr<thermostat::Thermostat>       _thermostat    = std::make_unique<thermostat::Thermostat>();
    std::unique_ptr<manostat::Manostat>           _manostat      = std::make_unique<manostat::Manostat>();
    std::unique_ptr<virial::Virial>               _virial        = std::make_unique<virial::VirialMolecular>();
    std::unique_ptr<resetKinetics::ResetKinetics> _resetKinetics = std::make_unique<resetKinetics::ResetKinetics>();

    std::unique_ptr<output::EnergyOutput>     _energyOutput  = std::make_unique<output::EnergyOutput>("default.en");
    std::unique_ptr<output::TrajectoryOutput> _xyzOutput     = std::make_unique<output::TrajectoryOutput>("default.xyz");
    std::unique_ptr<output::TrajectoryOutput> _velOutput     = std::make_unique<output::TrajectoryOutput>("default.vel");
    std::unique_ptr<output::TrajectoryOutput> _forceOutput   = std::make_unique<output::TrajectoryOutput>("default.force");
    std::unique_ptr<output::TrajectoryOutput> _chargeOutput  = std::make_unique<output::TrajectoryOutput>("default.chg");
    std::unique_ptr<output::LogOutput>        _logOutput     = std::make_unique<output::LogOutput>("default.log");
    std::unique_ptr<output::StdoutOutput>     _stdoutOutput  = std::make_unique<output::StdoutOutput>("stdout");
    std::unique_ptr<output::RstFileOutput>    _rstFileOutput = std::make_unique<output::RstFileOutput>("default.rst");
    std::unique_ptr<output::InfoOutput>       _infoOutput    = std::make_unique<output::InfoOutput>("default.info");

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

    output::EnergyOutput     &getEnergyOutput() { return *_energyOutput; }
    output::TrajectoryOutput &getXyzOutput() { return *_xyzOutput; }
    output::TrajectoryOutput &getVelOutput() { return *_velOutput; }
    output::TrajectoryOutput &getForceOutput() { return *_forceOutput; }
    output::TrajectoryOutput &getChargeOutput() { return *_chargeOutput; }
    output::LogOutput        &getLogOutput() { return *_logOutput; }
    output::StdoutOutput     &getStdoutOutput() { return *_stdoutOutput; }
    output::RstFileOutput    &getRstFileOutput() { return *_rstFileOutput; }
    output::InfoOutput       &getInfoOutput() { return *_infoOutput; }
};

#endif   // _ENGINE_HPP_