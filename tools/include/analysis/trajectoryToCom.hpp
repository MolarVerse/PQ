#ifndef _TRAJECTORYTOCOM_HPP_

#define _TRAJECTORYTOCOM_HPP_

#include "analysis.hpp"
#include "configurationReader.hpp"

class TrajectoryToCom : Analysis
{
    ConfigurationReader _configReader;

public:
    void setup() override;
    void run() override;
};

#endif // _TRAJECTORYTOCOM_HPP_