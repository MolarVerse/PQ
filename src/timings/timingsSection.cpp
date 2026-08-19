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

#include "timingsSection.hpp"

#include <chrono>

#include "constants/conversionFactors.hpp"

using namespace timings;

using Time     = std::chrono::time_point<std::chrono::high_resolution_clock>;
using Duration = std::chrono::duration<double>;
using std::chrono::duration_cast;

using ms = std::chrono::milliseconds;
using ns = std::chrono::nanoseconds;

/**
 * @brief Timings struct to store timing information
 *
 */
struct TimingsSection::Timings
{
    Time     start;
    Time     end;
    Duration totalTime    = Duration::zero();
    Duration lastStepTime = Duration::zero();
};

/**
 * @brief Construct a new Timings Section:: Timings Section object
 *
 * @param name
 */
TimingsSection::TimingsSection(const std::string_view name)
    : _name(name), _time(std::make_unique<Timings>())
{
}

/**
 * @brief Copy constructor for TimingsSection
 *
 * @param other
 */
TimingsSection::TimingsSection(const TimingsSection& other)
    : _name(other._name),
      _steps(other._steps),
      _time(other._time ? std::make_unique<Timings>(*other._time) : nullptr)
{
}

/**
 * @brief Copy assignment operator for TimingsSection
 *
 * @param other
 * @return TimingsSection&
 */
TimingsSection& TimingsSection::operator=(const TimingsSection& other)
{
    if (this != &other)
    {
        _time = other._time ? std::make_unique<Timings>(*other._time) : nullptr;
        _name = other._name;
        _steps = other._steps;
    }
    return *this;
}

TimingsSection::~TimingsSection()                                    = default;
TimingsSection::TimingsSection(TimingsSection&&) noexcept            = default;
TimingsSection& TimingsSection::operator=(TimingsSection&&) noexcept = default;

/**
 * @brief
 *
 */
void TimingsSection::beginTimer()
{
    _time->start = std::chrono::high_resolution_clock::now();
}

/**
 * @brief end the timer
 *
 */
void TimingsSection::endTimer()
{
    _time->end           = std::chrono::high_resolution_clock::now();
    _steps               = _steps + 1;
    _time->totalTime    += _time->end - _time->start;
    _time->lastStepTime  = _time->end - _time->start;
}

/**
 * @brief calculates the elapsed time in ms
 *
 */
double TimingsSection::calculateElapsedTime() const
{
    return static_cast<double>(duration_cast<ns>(_time->totalTime).count()) *
           constants::NS_TO_MS;
}

double TimingsSection::calculateAverageLoopTime() const
{
    auto time =
        static_cast<double>(duration_cast<ns>(_time->totalTime).count());

    time = time * constants::NS_TO_S / static_cast<double>(_steps);

    return time;
}

/**
 * @brief calculates the loop time in s
 *
 */
double TimingsSection::calculateLoopTime() const
{
    auto time =
        static_cast<double>(duration_cast<ns>(_time->lastStepTime).count());
    time = time * constants::NS_TO_S;

    return time;
}

/**
 * @brief get the name of the timings section
 *
 * @return std::string
 */
std::string TimingsSection::getName() const { return _name; }
