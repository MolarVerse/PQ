#ifndef _ANALYSIS_HPP_

#define _ANALYSIS_HPP_

#include <vector>
#include <string>

#include "frame.hpp"

class AnalysisRunner
{
protected:
    std::string _inputFilename;

    std::vector<std::string> _xyzFilenames;

    std::vector<Frame> _frames;

public:
    AnalysisRunner() = default;

    virtual void setup() = 0;
    virtual void run() = 0;
};

#endif // _ANALYSIS_HPP_