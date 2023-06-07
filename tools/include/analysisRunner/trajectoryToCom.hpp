#ifndef _TRAJECTORYTOCOM_HPP_

#define _TRAJECTORYTOCOM_HPP_

#include "analysisRunner.hpp"
#include "configurationReader.hpp"

class TrajToCom : public AnalysisRunner
{
    ConfigurationReader _configReader;

public:
    using AnalysisRunner::AnalysisRunner;

    void setup() override;
    void run() override;
};

#endif // _TRAJECTORYTOCOM_HPP_