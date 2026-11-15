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

#ifndef SETTINGS__TIMINGS_SETTINGS_HPP
#define SETTINGS__TIMINGS_SETTINGS_HPP

#include <cstddef>

#include "setting.hpp"
#include "unit_types.hpp"

/**
 * @class TimingsSettings
 *
 * @brief static class to store settings of timings
 *
 */
class TimingsSettings
{
   private:
    Setting<Time, "Time Step">         _timeStep;
    Setting<size_t, "Number of Steps"> _numberOfSteps;
    static inline size_t               _stepCount = 0;

   public:
    /********************
     * standard setters *
     ********************/

    void        setTimeStepByUser(const Time& timeStep);
    static void setStepCount(size_t stepCount);
    void        setNumberOfStepsByUser(size_t numberOfSteps);

    /********************
     * standard getters *
     ********************/

    [[nodiscard]] const Time&   getTimeStep() const;
    [[nodiscard]] static size_t getStepCount();
    [[nodiscard]] size_t        getNumberOfSteps() const;
};

#endif   // SETTINGS__TIMINGS_SETTINGS_HPP
