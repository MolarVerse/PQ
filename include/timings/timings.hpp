/*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

#ifndef _TIMINGS_HPP_

#define _TIMINGS_HPP_

#include <chrono>    // IWYU pragma: keep for time_point, milliseconds, nanoseconds
#include <cstddef>   // for size_t

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
