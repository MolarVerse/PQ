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

#ifndef _GLOBAL_TIMER_HPP_

#define _GLOBAL_TIMER_HPP_

#include "timer.hpp"

namespace timings
{
    class GlobalTimer
    {
       private:
        Timer _simulationTimer;

        std::vector<Timer> _timers;

       public:
        void addSimulationTimer(const Timer &simulationTimer);

        [[nodiscard]] double calculateLoopTime() const;
        [[nodiscard]] double calculateElapsedTime() const;

        void sortTimers();

        void startSimulationTimer();
        void stopSimulationTimer();

        void addTimer(const Timer &timer);

        [[nodiscard]] const std::vector<Timer> &getTimers() const;
    };
}   // namespace timings

#endif   // _GLOBAL_TIMER_HPP_