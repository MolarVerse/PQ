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

#ifndef _TIMINGS_SECTION_GUARD_HPP_

#define _TIMINGS_SECTION_GUARD_HPP_

#include <string>        // for string
#include <string_view>   // for string_view

namespace timings
{
    class Timer;   // forward declaration

    /**
     * @class TimingsSectionGuard
     *
     * @brief RAII guard that starts a named timings section on
     * construction and stops it on destruction - including on early
     * return or when unwinding due to an exception.
     *
     * @details
     *  Intentionally neither copyable nor movable: it represents exactly
     *  one active measurement scope tied to the stack frame that created
     *  it. Construct it with `Timer::scoped(name)` and bind it to a
     *  local variable, e.g.:
     *
     *      auto _ = someTimer.scoped("nonCoulomb");
     *
     */
    class TimingsSectionGuard
    {
       private:
        Timer&      _timer;
        std::string _name;

       public:
        explicit TimingsSectionGuard(Timer& timer, const std::string_view name);
        ~TimingsSectionGuard();

        TimingsSectionGuard(const TimingsSectionGuard&)            = delete;
        TimingsSectionGuard& operator=(const TimingsSectionGuard&) = delete;
        TimingsSectionGuard(TimingsSectionGuard&&)                 = delete;
        TimingsSectionGuard& operator=(TimingsSectionGuard&&)      = delete;
    };

}   // namespace timings

#endif   // _TIMINGS_SECTION_GUARD_HPP_
