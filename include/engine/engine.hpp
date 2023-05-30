#ifndef _ENGINE_H_

#define _ENGINE_H_

#include <vector>
#include <memory>

#include "settings.hpp"
#include "timings.hpp"
#include "output.hpp"
#include "physicalData.hpp"
#include "simulationBox.hpp"
#include "integrator.hpp"
#include "celllist.hpp"
#include "potential.hpp"
#include "thermostat.hpp"
#include "manostat.hpp"
#include "virial.hpp"

/**
 * @class Engine
 *
 * @brief Contains all the information needed to run the simulation
 *
 */
class Engine
{
private:
    Settings _settings;
    Timings _timings;
    CellList _cellList;

    SimulationBox _simulationBox;

    PhysicalData _physicalData;
    PhysicalData _averagePhysicalData;

public:
    std::unique_ptr<Integrator> _integrator;
    std::unique_ptr<Potential> _potential = std::make_unique<PotentialBruteForce>();
    std::unique_ptr<Thermostat> _thermostat = std::make_unique<Thermostat>();
    std::unique_ptr<Manostat> _manostat = std::make_unique<Manostat>();
    std::unique_ptr<Virial> _virial = std::make_unique<VirialMolecular>();

    std::unique_ptr<EnergyOutput> _energyOutput = std::make_unique<EnergyOutput>("default.en");
    std::unique_ptr<TrajectoryOutput> _xyzOutput = std::make_unique<TrajectoryOutput>("default.xyz");
    std::unique_ptr<TrajectoryOutput> _velOutput = std::make_unique<TrajectoryOutput>("default.vel");
    std::unique_ptr<TrajectoryOutput> _forceOutput = std::make_unique<TrajectoryOutput>("default.force");
    std::unique_ptr<LogOutput> _logOutput = std::make_unique<LogOutput>("default.log");
    std::unique_ptr<StdoutOutput> _stdoutOutput = std::make_unique<StdoutOutput>("stdout");
    std::unique_ptr<RstFileOutput> _rstFileOutput = std::make_unique<RstFileOutput>("default.rst");
    std::unique_ptr<ChargeOutput> _chargeOutput = std::make_unique<ChargeOutput>("default.chg");
    std::unique_ptr<InfoOutput> _infoOutput = std::make_unique<InfoOutput>("default.info");

    void run();
    void takeStep();

    // standard getter and setters
    Settings &getSettings() { return _settings; };
    Timings &getTimings() { return _timings; };
    CellList &getCellList() { return _cellList; };
    SimulationBox &getSimulationBox() { return _simulationBox; };
    PhysicalData &getPhysicalData() { return _physicalData; };
};

#endif