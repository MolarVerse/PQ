#ifndef _TIMINGS_H_

#define _TIMINGS_H_

/**
 * @class Timings
 *
 * @brief Stores all timings information
 *
 * @details
 *
 *  stores as well internal simulation timings
 *  as well as all timings corresponding to
 *  execution time
 *
 */
class Timings
{
private:
    int _stepCount = 0;
    int _timestep = 0;

public:
    int getStepCount() const { return _stepCount; };
    void setStepCount(int stepCount);

    int getTimestep() const { return _timestep; };
    void setTimestep(int timestep) { _timestep = timestep; };
};

#endif
