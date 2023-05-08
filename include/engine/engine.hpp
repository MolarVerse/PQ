#ifndef _ENGINE_H_

#define _ENGINE_H_

#include <vector>

#include "settings.hpp"
#include "jobtype.hpp"
#include "output.hpp"
#include "simulationBox.hpp"
#include "integrator.hpp"

/**
 * @class Engine
 *
 * @brief Contains all the information needed to run the simulation
 *
 */
class Engine
{
public:
    Engine();
    Engine(const Engine &);

    Settings _settings;
    JobType _jobType;

    EnergyOutput _energyOutput = EnergyOutput();
    TrajectoryOutput _xyzOutput = TrajectoryOutput();
    TrajectoryOutput _velOutput = TrajectoryOutput();
    TrajectoryOutput _forceOutput = TrajectoryOutput();
    LogOutput _logOutput = LogOutput();
    StdoutOutput _stdoutOutput = StdoutOutput();
    RstFileOutput _rstFileOutput = RstFileOutput();
    ChargeOutput _chargeOutput = ChargeOutput();
    InfoOutput _infoOutput = InfoOutput();

    SimulationBox _simulationBox;
    Integrator _integrator;
};

#endif