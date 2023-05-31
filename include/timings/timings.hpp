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
    double _timestep = 0;
    int _numberOfSteps = 0;

public:
    void setNumberOfSteps(int numberOfSteps);

    // standard getter and setters
    [[nodiscard]] int getStepCount() const { return _stepCount; };
    void setStepCount(int stepCount);

    [[nodiscard]] double getTimestep() const { return _timestep; };
    void setTimestep(const double timestep) { _timestep = timestep; };

    [[nodiscard]] int getNumberOfSteps() const { return _numberOfSteps; };
};

#endif
