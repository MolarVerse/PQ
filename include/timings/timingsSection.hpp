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

#ifndef _TIMINGS_SECTION_HPP_

#define _TIMINGS_SECTION_HPP_

#include <cstddef>   // for size_t
#include <memory>    // for unique_ptr
#include <string>    // for string, allocator, basic_string

namespace timings
{
    /**
     * @class TimingsSection
     *
     * @brief Stores all timings information
     *
     * @details
     *  stores internal simulation timings
     *  as well as all timings corresponding to
     *  execution time
     *
     */
    class TimingsSection
    {
       private:
        std::string _name;
        size_t      _steps = 0;

        struct Timings;
        std::unique_ptr<Timings> _time;

       public:
        explicit TimingsSection(const std::string_view name);
        ~TimingsSection();

        TimingsSection(const TimingsSection&);
        TimingsSection& operator=(const TimingsSection&);
        TimingsSection(TimingsSection&&);
        TimingsSection& operator=(TimingsSection&&);

        void beginTimer();
        void endTimer();

        [[nodiscard]] double calculateElapsedTime() const;
        [[nodiscard]] double calculateLoopTime() const;
        [[nodiscard]] double calculateAverageLoopTime() const;

        [[nodiscard]] std::string getName() const;
    };

}   // namespace timings

#endif   // _TIMINGS_SECTION_HPP_
