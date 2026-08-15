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

#include "singleton.hpp"
#include "timer.hpp"
#include "timerId.hpp"
#include "timingsSectionGuard.hpp"

namespace timings
{
    class GlobalTimer : public Singleton<GlobalTimer>
    {
       private:
        std::array<Timer, TimerIdMeta::size> _timers{};

       public:
        [[nodiscard]] double calculateLoopTime() const;
        [[nodiscard]] double calculateElapsedTime() const;

        std::vector<Timer> sortTimers() const;

        void stopSimulationTimer();
        void stopAndRestartSimulationTimer();

        [[nodiscard]]
        TimingsSectionGuard scoped(TimerId id, const std::string &sectionName);

       private:
        friend class Singleton<GlobalTimer>;
        GlobalTimer();

        [[nodiscard]] Timer       &_getSimulationTimer();
        [[nodiscard]] const Timer &_getSimulationTimer() const;

        [[nodiscard]] Timer       &_getTimer(TimerId id);
        [[nodiscard]] const Timer &_getTimer(TimerId id) const;
    };
}   // namespace timings

/**
 * @brief Scoped timer for a specific section of code.
 *
 * This function creates a scoped timer for a specific section of code. It
 * starts the timer when the function is called and stops it when the returned
 * TimingsSectionGuard object goes out of scope.
 *
 * @param id The TimerId for the section of code being timed.
 * @param sectionName A descriptive name for the section of code being timed.
 *
 * @return A TimingsSectionGuard object that manages the timing of the specified
 *         section of code.
 */
[[nodiscard]]
inline timings::TimingsSectionGuard scopedTimer(
    TimerId            id,
    const std::string &sectionName
)
{
    return timings::GlobalTimer::get().scoped(id, sectionName);
}

#endif   // _GLOBAL_TIMER_HPP_
