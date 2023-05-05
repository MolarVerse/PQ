#ifndef _ENGINE_H_

#define _ENGINE_H_

#include <vector>

#include "settings.hpp"
#include "jobtype.hpp"
#include "output.hpp"
#include "simulationBox.hpp"

class Engine
{
public:
    Engine() = default;
    ~Engine() = default;

    Settings _settings;
    JobType _jobType;
    std::vector<Output> _output = {StdoutOutput()};
    SimulationBox _simulationBox;
};

#endif