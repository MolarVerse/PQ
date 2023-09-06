#ifndef _TIMINGS_HPP_

#define _TIMINGS_HPP_

#include <bits/chrono.h>   // for duration_cast, high_resolution_clock, operator-
#include <cstddef>         // for size_t

namespace timings
{
    using Time = std::chrono::time_point<std::chrono::high_resolution_clock>;
    using ms   = std::chrono::milliseconds;
    using ns   = std::chrono::nanoseconds;

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
        size_t _stepCount = 0;

        Time _start;
        Time _end;

      public:
        void beginTimer() { _start = std::chrono::high_resolution_clock::now(); }
        void endTimer() { _end = std::chrono::high_resolution_clock::now(); }

        [[nodiscard]] long   calculateElapsedTime() const { return std::chrono::duration_cast<ms>(_end - _start).count(); }
        [[nodiscard]] double calculateLoopTime(const size_t numberOfSteps)
        {
            _end = std::chrono::high_resolution_clock::now();
            return double(std::chrono::duration_cast<ns>(_end - _start).count()) * 1e-9 / double(numberOfSteps);
        }

        /********************************
         * standard getters and setters *
         ********************************/

        [[nodiscard]] size_t getStepCount() const { return _stepCount; }

        void setStepCount(const size_t stepCount) { _stepCount = stepCount; }
    };

}   // namespace timings

#endif   // _TIMINGS_HPP_
