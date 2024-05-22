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

#include <chrono>   // IWYU pragma: keep for time_point, milliseconds, nanoseconds
#include <cstddef>   // for size_t
#include <string>    // for string

#include "timingsSection.hpp"   // for TimingsManager

namespace timings
{
    using Time = std::chrono::time_point<std::chrono::high_resolution_clock>;
    using Duration = std::chrono::duration<double>;
    using ms       = std::chrono::milliseconds;
    using ns       = std::chrono::nanoseconds;

    using namespace std::chrono;

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
        std::string _name = "DefaultTimings";

        std::vector<TimingsSection> _timingDetails;

       public:
        explicit Timer(const std::string_view);

        Timer()  = default;
        ~Timer() = default;

        [[nodiscard]] std::vector<TimingsSection> getTimingDetails() const;
        [[nodiscard]] double                      calculateElapsedTime() const;
        [[nodiscard]] double                      calculateLoopTime() const;

        [[nodiscard]] size_t findTimingsSectionIndex(const std::string_view name
        ) const;

        void startTimingsSection();
        void startTimingsSection(const std::string_view name);
        void stopTimingsSection();
        void stopTimingsSection(const std::string_view name);

        void sortTimingsSections();

        [[nodiscard]] TimingsSection getTimingsSection(
            const std::string_view name
        ) const;

        /********************************
         * standard getters and setters *
         ********************************/

        void setTimerName(const std::string_view name) { _name = name; }

        [[nodiscard]] std::string getTimerName() const { return _name; }
        [[nodiscard]] Timer       getTimer() const { return *this; }
    };

}   // namespace timings

#endif   // _TIMER_HPP_
