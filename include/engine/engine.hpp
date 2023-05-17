#ifndef _ENGINE_H_

#define _ENGINE_H_

#include <vector>
#include <memory>

#include "settings.hpp"
#include "jobtype.hpp"
#include "output.hpp"
#include "outputData.hpp"
#include "simulationBox.hpp"
#include "integrator.hpp"
#include "celllist.hpp"

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
    std::unique_ptr<JobType> _jobType;
    OutputData _outputData;
    Integrator _integrator;
    CellList _cellList;

    std::unique_ptr<EnergyOutput> _energyOutput = std::make_unique<EnergyOutput>(EnergyOutput("default.en"));
    std::unique_ptr<TrajectoryOutput> _xyzOutput = std::make_unique<TrajectoryOutput>(TrajectoryOutput("default.xyz"));
    std::unique_ptr<TrajectoryOutput> _velOutput = std::make_unique<TrajectoryOutput>(TrajectoryOutput("default.vel"));
    std::unique_ptr<TrajectoryOutput> _forceOutput = std::make_unique<TrajectoryOutput>(TrajectoryOutput("default.force"));
    std::unique_ptr<LogOutput> _logOutput = std::make_unique<LogOutput>(LogOutput("default.log"));
    std::unique_ptr<StdoutOutput> _stdoutOutput = std::make_unique<StdoutOutput>(StdoutOutput("stdout"));
    std::unique_ptr<RstFileOutput> _rstFileOutput = std::make_unique<RstFileOutput>(RstFileOutput("default.rst"));
    std::unique_ptr<ChargeOutput> _chargeOutput = std::make_unique<ChargeOutput>(ChargeOutput("default.chg"));
    std::unique_ptr<InfoOutput> _infoOutput = std::make_unique<InfoOutput>(InfoOutput("default.info"));

    SimulationBox &getSimulationBox() { return _simulationBox; };
    void calculateMomentum(SimulationBox &, OutputData &);
};

#endif