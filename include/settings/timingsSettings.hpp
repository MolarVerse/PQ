#ifndef _TIMINGS_SETTINGS_HPP_

#define _TIMINGS_SETTINGS_HPP_

#include <cstddef>   // for size_t

namespace settings
{
    /**
     * @class TimingsSettings
     *
     * @brief static class to store settings of timings
     *
     */
    class TimingsSettings
    {
      private:
        static inline double _timeStep;
        static inline size_t _numberOfSteps;

      public:
        static void setTimeStep(const double timeStep) { _timeStep = timeStep; }
        static void setNumberOfSteps(const size_t numberOfSteps) { _numberOfSteps = numberOfSteps; }

        [[nodiscard]] static double getTimeStep() { return _timeStep; }
        [[nodiscard]] static size_t getNumberOfSteps() { return _numberOfSteps; }
    };
}   // namespace settings

#endif   // _TIMINGS_SETTINGS_HPP_