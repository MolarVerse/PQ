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

#ifndef _TIMER_HPP_

#define _TIMER_HPP_

#include <cstddef>   // for size_t
#include <string>    // for string
#include <vector>    // for vector

#include "timerId.hpp"
#include "timingsSection.hpp"   // for TimingsManager
#include "timingsSectionGuard.hpp"

namespace timings
{

    /**
     * @class Timer
     *
     * @brief Stores all timings information
     *
     * @details
     *  stores internal simulation timings
     *  as well as all timings corresponding to
     *  execution time
     *
     */
    class Timer
    {
       protected:
        TimerId _id = TimerId::DefaultTimings;

        std::vector<TimingsSection> _timingDetails;

       public:
        explicit Timer(TimerId id);
        Timer() = default;

        [[nodiscard]]
        std::vector<TimingsSection> getTimingDetails() const;

        [[nodiscard]] double calculateElapsedTime() const;
        [[nodiscard]] double calculateLoopTime() const;

        [[nodiscard]] size_t findTimingsSectionIndex(
            const std::string_view name
        ) const;

        void startTimingsSection();
        void stopTimingsSection();

        void sortTimingsSections();

        /********************
         * standard setters *
         ********************/

        void setTimerId(TimerId id);

        /********************
         * standard getters *
         ********************/

        [[nodiscard]] TimingsSection getTimingsSection(
            const std::string_view
        ) const;

        [[nodiscard]] std::string getTimerName() const;
        [[nodiscard]] Timer       getTimer() const;

        [[nodiscard]] TimingsSectionGuard scoped(const std::string_view name);

       private:
        friend class TimingsSectionGuard;

        void startTimingsSection(const std::string_view name);
        void stopTimingsSection(const std::string_view name);
    };

}   // namespace timings

#endif   // _TIMER_HPP_
