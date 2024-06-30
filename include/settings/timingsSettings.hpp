/*****************************************************************************
<GPL_HEADER>

    PQ
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
        static inline size_t _stepCount = 0;

        static inline bool _isTimeStepSet = false;

       public:
        TimingsSettings()  = default;
        ~TimingsSettings() = default;

        /********************
         * standard setters *
         ********************/

        static void setTimeStep(const double timeStep);
        static void setStepCount(const size_t stepCount);
        static void setNumberOfSteps(const size_t numberOfSteps);

        /********************
         * standard setters *
         ********************/

        [[nodiscard]] static double getTimeStep();
        [[nodiscard]] static size_t getStepCount();
        [[nodiscard]] static size_t getNumberOfSteps();
        [[nodiscard]] static bool   isTimeStepSet();
    };
}   // namespace settings

#endif   // _TIMINGS_SETTINGS_HPP_