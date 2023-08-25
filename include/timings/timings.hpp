#ifndef _TIMINGS_HPP_

#define _TIMINGS_HPP_

#include <bits/chrono.h>   // for duration_cast, high_resolution_clock, operator-
#include <cstddef>         // for size_t

namespace timings
{
    using Time = std::chrono::time_point<std::chrono::high_resolution_clock>;
    using ms   = std::chrono::milliseconds;

    /**
     * @class Timings
     *
     * @brief Stores all timings information
     *
     * @details
     *  stores internal simulation timings
     *  as well as all timings corresponding to
     *  execution time
     *
     */
    class Timings
    {
      private:
        size_t _stepCount     = 0;
        size_t _numberOfSteps = 0;
        double _timestep      = 0.0;

        Time _start;
        Time _end;

      public:
        void beginTimer() { _start = std::chrono::high_resolution_clock::now(); }
        void endTimer() { _end = std::chrono::high_resolution_clock::now(); }

        [[nodiscard]] long calculateElapsedTime() const { return std::chrono::duration_cast<ms>(_end - _start).count(); }

        /********************************
         * standard getters and setters *
         ********************************/

        [[nodiscard]] size_t getStepCount() const { return _stepCount; }
        [[nodiscard]] size_t getNumberOfSteps() const { return _numberOfSteps; }
        [[nodiscard]] double getTimestep() const { return _timestep; }

        void setStepCount(const size_t stepCount) { _stepCount = stepCount; }
        void setTimestep(const double timestep) { _timestep = timestep; }
        void setNumberOfSteps(const size_t numberOfSteps) { _numberOfSteps = numberOfSteps; }
    };

}   // namespace timings

#endif   // _TIMINGS_HPP_
