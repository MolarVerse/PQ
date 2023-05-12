#ifndef _ENGINE_H_

#define _ENGINE_H_

#include <vector>
#include <memory>

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
private:
    SimulationBox _simulationBox;

public:
    // Engine() {};
    //  Engine(const Engine &);

    Settings _settings;
    JobType _jobType;

    std::shared_ptr<EnergyOutput> _energyOutput = std::make_shared<EnergyOutput>(EnergyOutput("default.en"));
    std::shared_ptr<TrajectoryOutput> _xyzOutput = std::make_shared<TrajectoryOutput>(TrajectoryOutput("default.xyz"));
    std::shared_ptr<TrajectoryOutput> _velOutput = std::make_shared<TrajectoryOutput>(TrajectoryOutput("default.vel"));
    std::shared_ptr<TrajectoryOutput> _forceOutput = std::make_shared<TrajectoryOutput>(TrajectoryOutput("default.force"));
    std::shared_ptr<LogOutput> _logOutput = std::make_shared<LogOutput>(LogOutput("default.log"));
    std::shared_ptr<StdoutOutput> _stdoutOutput = std::make_shared<StdoutOutput>(StdoutOutput("stdout"));
    std::shared_ptr<RstFileOutput> _rstFileOutput = std::make_shared<RstFileOutput>(RstFileOutput("default.rst"));
    std::shared_ptr<ChargeOutput> _chargeOutput = std::make_shared<ChargeOutput>(ChargeOutput("default.chg"));
    std::shared_ptr<InfoOutput> _infoOutput = std::make_shared<InfoOutput>(InfoOutput("default.info"));

    SimulationBox &getSimulationBox() { return _simulationBox; };

    Integrator _integrator;
};

#endif