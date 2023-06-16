#ifndef _ENGINE_HPP_

#define _ENGINE_HPP_

#include "celllist.hpp"
#include "energyOutput.hpp"
#include "infoOutput.hpp"
#include "integrator.hpp"
#include "manostat.hpp"
#include "output.hpp"
#include "physicalData.hpp"
#include "potential.hpp"
#include "settings.hpp"
#include "simulationBox.hpp"
#include "thermostat.hpp"
#include "timings.hpp"
#include "trajectoryOutput.hpp"
#include "virial.hpp"

#include <memory>
#include <vector>

/**
 * @class Engine
 *
 * @brief Contains all the information needed to run the simulation
 *
 */
class Engine
{
  private:
    size_t _step = 1;

    Settings                _settings;
    Timings                 _timings;
    simulationBox::CellList _cellList;

    simulationBox::SimulationBox _simulationBox;

    PhysicalData _physicalData;
    PhysicalData _averagePhysicalData;

  public:
    std::unique_ptr<Integrator>             _integrator;
    std::unique_ptr<Potential>              _potential  = std::make_unique<PotentialBruteForce>();
    std::unique_ptr<thermostat::Thermostat> _thermostat = std::make_unique<thermostat::Thermostat>();
    std::unique_ptr<Manostat>               _manostat   = std::make_unique<Manostat>();
    std::unique_ptr<virial::Virial>         _virial     = std::make_unique<virial::VirialMolecular>();

    std::unique_ptr<EnergyOutput>     _energyOutput  = std::make_unique<EnergyOutput>("default.en");
    std::unique_ptr<TrajectoryOutput> _xyzOutput     = std::make_unique<TrajectoryOutput>("default.xyz");
    std::unique_ptr<TrajectoryOutput> _velOutput     = std::make_unique<TrajectoryOutput>("default.vel");
    std::unique_ptr<TrajectoryOutput> _forceOutput   = std::make_unique<TrajectoryOutput>("default.force");
    std::unique_ptr<TrajectoryOutput> _chargeOutput  = std::make_unique<TrajectoryOutput>("default.chg");
    std::unique_ptr<LogOutput>        _logOutput     = std::make_unique<LogOutput>("default.log");
    std::unique_ptr<StdoutOutput>     _stdoutOutput  = std::make_unique<StdoutOutput>("stdout");
    std::unique_ptr<RstFileOutput>    _rstFileOutput = std::make_unique<RstFileOutput>("default.rst");
    std::unique_ptr<InfoOutput>       _infoOutput    = std::make_unique<InfoOutput>("default.info");

    void run();
    void takeStep();
    void writeOutput();

    // standard getter and setters
    [[nodiscard]] Settings                     &getSettings() { return _settings; }
    [[nodiscard]] Timings                      &getTimings() { return _timings; }
    [[nodiscard]] simulationBox::CellList      &getCellList() { return _cellList; }
    [[nodiscard]] simulationBox::SimulationBox &getSimulationBox() { return _simulationBox; }
    [[nodiscard]] PhysicalData                 &getPhysicalData() { return _physicalData; }
    [[nodiscard]] PhysicalData                 &getAveragePhysicalData() { return _averagePhysicalData; }
};

#endif   // _ENGINE_HPP_