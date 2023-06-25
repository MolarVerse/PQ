#ifndef _TIMINGS_H_

#define _TIMINGS_H_

#include <chrono>
#include <cstddef>

namespace timings
{
    using Time = std::chrono::time_point<std::chrono::high_resolution_clock>;
    class Timings;
}   // namespace timings

/**
 * @class Timings
 *
 * @brief Stores all timings information
 *
 * @details
 *  stores as well internal simulation timings
 *  as well as all timings corresponding to
 *  execution time
 *
 */
class timings::Timings
{
  private:
    size_t _stepCount     = 0;
    double _timestep      = 0;
    size_t _numberOfSteps = 0;

    Time _start;
    Time _end;

  public:
    void beginTimer() { _start = std::chrono::high_resolution_clock::now(); }
    void endTimer() { _end = std::chrono::high_resolution_clock::now(); }

    long calculateElapsedTime() const { return std::chrono::duration_cast<std::chrono::nanoseconds>(_end - _start).count(); }

    /********************************
     * standard getters and setters *
     ********************************/

    size_t getStepCount() const { return _stepCount; }
    size_t getNumberOfSteps() const { return _numberOfSteps; }
    double getTimestep() const { return _timestep; }

    void setStepCount(const size_t stepCount) { _stepCount = stepCount; }
    void setTimestep(const double timestep) { _timestep = timestep; }
    void setNumberOfSteps(const size_t numberOfSteps) { _numberOfSteps = numberOfSteps; }
};

#endif
