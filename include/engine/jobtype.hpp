#ifndef _JOB_TYPE_H_

#define _JOB_TYPE_H_

#include <string>

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
};

#endif
