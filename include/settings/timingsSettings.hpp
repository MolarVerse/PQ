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
        TimingsSettings()  = default;
        ~TimingsSettings() = default;

        static void setTimeStep(const double timeStep) { _timeStep = timeStep; }
        static void setNumberOfSteps(const size_t numberOfSteps) { _numberOfSteps = numberOfSteps; }

        [[nodiscard]] static double getTimeStep() { return _timeStep; }
        [[nodiscard]] static size_t getNumberOfSteps() { return _numberOfSteps; }
    };
}   // namespace settings

#endif   // _TIMINGS_SETTINGS_HPP_