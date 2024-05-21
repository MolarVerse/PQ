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

#ifndef _TIMINGS_HPP_

#define _TIMINGS_HPP_

#include <chrono>   // IWYU pragma: keep for time_point, milliseconds, nanoseconds
#include <cstddef>   // for size_t
#include <string>    // for string

#include "timingsManager.hpp"   // for TimingsManager

namespace timings
{
    using Time = std::chrono::time_point<std::chrono::high_resolution_clock>;
    using Duration = std::chrono::duration<double>;
    using ms       = std::chrono::milliseconds;
    using ns       = std::chrono::nanoseconds;

    using namespace std::chrono;

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
       protected:
        std::string _name = "DefaultTimings";

        std::vector<TimingsManager> _timingDetails;

       public:
        explicit Timings(const std::string_view);

        Timings()  = default;
        ~Timings() = default;

        [[nodiscard]] long   calculateElapsedTime() const;
        [[nodiscard]] double calculateLoopTime();

        [[nodiscard]] size_t findTimeManagerIndex(const std::string_view) const;

        void startTimeManager(const std::string_view name);
        void stopTimeManager(const std::string_view name);

        [[nodiscard]] TimingsManager getTimingsManager(
            const std::string_view name
        ) const;

        /********************************
         * standard getters and setters *
         ********************************/

        [[nodiscard]] Timings getTimings() const { return *this; }
    };

}   // namespace timings

#endif   // _TIMINGS_HPP_
