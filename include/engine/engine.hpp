#ifndef _ENGINE_H_

#define _ENGINE_H_

#include <vector>
#include <memory>

#include "settings.hpp"
#include "output.hpp"
#include "outputData.hpp"
#include "simulationBox.hpp"
#include "integrator.hpp"
#include "celllist.hpp"
#include "potential.hpp"

/**
 * @class Engine
 *
 * @brief Contains all the information needed to run the simulation
 *
 */
class Engine
{
private:
    SimulationBox _simulationBox;

public:
    Settings _settings;
    std::unique_ptr<Potential> _potential = std::make_unique<PotentialBruteForce>();
    OutputData _outputData;
    Integrator _integrator;
    CellList _cellList;

    std::unique_ptr<EnergyOutput> _energyOutput = std::make_unique<EnergyOutput>("default.en");
    std::unique_ptr<TrajectoryOutput> _xyzOutput = std::make_unique<TrajectoryOutput>("default.xyz");
    std::unique_ptr<TrajectoryOutput> _velOutput = std::make_unique<TrajectoryOutput>("default.vel");
    std::unique_ptr<TrajectoryOutput> _forceOutput = std::make_unique<TrajectoryOutput>("default.force");
    std::unique_ptr<LogOutput> _logOutput = std::make_unique<LogOutput>("default.log");
    std::unique_ptr<StdoutOutput> _stdoutOutput = std::make_unique<StdoutOutput>("stdout");
    std::unique_ptr<RstFileOutput> _rstFileOutput = std::make_unique<RstFileOutput>("default.rst");
    std::unique_ptr<ChargeOutput> _chargeOutput = std::make_unique<ChargeOutput>("default.chg");
    std::unique_ptr<InfoOutput> _infoOutput = std::make_unique<InfoOutput>("default.info");

    SimulationBox &getSimulationBox() { return _simulationBox; };
    void calculateMomentum(SimulationBox &, OutputData &) const;
};

#endif