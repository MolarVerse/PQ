#ifndef _JOB_TYPE_H_

#define _JOB_TYPE_H_

#include <string>

#include "simulationBox.hpp"
#include "outputData.hpp"

/**
 * @class JobType
 *
 * @brief base class for job types
 *
 */
class JobType
{
protected:
    std::string _jobType;

public:
    std::string getJobType() const { return _jobType; }
    void setJobType(std::string_view jobType) { _jobType = jobType; };

    virtual void calculateForces(SimulationBox &, OutputData &) = 0;
    void calcCoulomb(double, double, double, double &, double &, double, double force_cutof);
    void calcNonCoulomb(std::vector<double> &, double, double, double &, double &, double, double);
};

/**
 * @class MMMD
 *
 * @brief Molecular Mechanics Molecular Dynamics
 *
 * @details inherits from JobType
 *
 */
class MMMD : public JobType
{
public:
    MMMD() { _jobType = "MMMD"; };

    void calculateForces(SimulationBox &, OutputData &) override;
};

#endif
