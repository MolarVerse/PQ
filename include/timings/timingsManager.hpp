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

#ifndef _TIMINGS_MANAGER_HPP_

#define _TIMINGS_MANAGER_HPP_

#include <chrono>   // IWYU pragma: keep for time_point, milliseconds, nanoseconds
#include <cstddef>   // for size_t

namespace timings
{
    using Time = std::chrono::time_point<std::chrono::high_resolution_clock>;
    using Duration = std::chrono::duration<double>;
    using ms       = std::chrono::milliseconds;
    using ns       = std::chrono::nanoseconds;

    using namespace std::chrono;

    class TimingsManager
    {
       private:
        std::string _name;
        size_t      _steps = 0;

        Time     _start;
        Time     _end;
        Duration _totalTime;

       public:
        explicit TimingsManager(const std::string_view name) : _name(name) {}

        void endTimer();

        [[nodiscard]] long   calculateElapsedTime() const;
        [[nodiscard]] double calculateLoopTime() const;

        [[nodiscard]] std::string getName() const { return _name; }

        void beginTimer() { _start = high_resolution_clock::now(); }
    };

}   // namespace timings

#endif   // _TIMINGS_MANAGER_HPP_
