#ifndef _ENGINE_H_

#define _ENGINE_H_

#include <vector>

#include "settings.hpp"
#include "timings.hpp"
#include "jobtype.hpp"
#include "output.hpp"

class Engine
{
public:
    Engine() = default;
    ~Engine() = default;

    Settings _settings;
    Timings _timings;
    JobType _jobType;
    std::vector<Output> _output = {StdoutOutput()};
};

#endif