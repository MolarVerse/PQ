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

#include "timingsSectionGuard.hpp"

#include "timer.hpp"

namespace timings
{
    /**
     * @brief Construct a new Timings Section Guard:: Timings Section Guard
     * object
     *
     * @param timer
     * @param name
     */
    TimingsSectionGuard::TimingsSectionGuard(
        Timer&                 timer,
        const std::string_view name
    )
        : _timer(timer), _name(name)
    {
        _timer.startTimingsSection(_name);
    }

    /**
     * @brief Destroy the Timings Section Guard:: Timings Section Guard object
     *
     */
    TimingsSectionGuard::~TimingsSectionGuard()
    {
        _timer.stopTimingsSection(_name);
    }
}   // namespace timings
