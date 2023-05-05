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
    Engine() = default;
    ~Engine() = default;

    Settings _settings;
    JobType _jobType;
    std::vector<Output> _output = {StdoutOutput()};
    SimulationBox _simulationBox;
    Integrator _integrator;
};

#endif