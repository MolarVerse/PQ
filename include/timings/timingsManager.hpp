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

    class TimingsManager
    {
       private:
        std::string _name;
        size_t      _steps = 0;

        Time     _start;
        Time     _end;
        Duration _totalTime;

       public:
        explicit TimingsManager(const std::string& name) : _name(name) {}

        [[nodiscard]] std::string getName() const { return _name; }

        void beginTimer()
        {
            _start = std::chrono::high_resolution_clock::now();
        }

        void endTimer()
        {
            _end        = std::chrono::high_resolution_clock::now();
            _steps      = _steps + 1;
            _totalTime += _end - _start;
        }

        [[nodiscard]] long calculateElapsedTime() const
        {
            return std::chrono::duration_cast<ms>(_totalTime).count();
        }

        [[nodiscard]] double calculateLoopTime() const
        {
            return double(std::chrono::duration_cast<ns>(_totalTime).count()) *
                   1e-9 / double(_steps);
        }
    };

}   // namespace timings

#endif   // _TIMINGS_MANAGER_HPP_
