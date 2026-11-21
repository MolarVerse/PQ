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

#include "units.hpp"

namespace timings
{

    using ChronoTime =
        std::chrono::time_point<std::chrono::high_resolution_clock>;
    using ChronoDuration = std::chrono::duration<double>;

    /**
     * @brief Timings struct to store timing information
     *
     */
    struct TimingsSection::Timings
    {
        ChronoTime     start;
        ChronoTime     end;
        ChronoDuration totalTime    = ChronoDuration::zero();
        ChronoDuration lastStepTime = ChronoDuration::zero();
    };

    /**
     * @brief Construct a new Timings Section:: Timings Section object
     *
     * @param name
     */
    TimingsSection::TimingsSection(std::string_view name)
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
            _time =
                other._time ? std::make_unique<Timings>(*other._time) : nullptr;
            _name  = other._name;
            _steps = other._steps;
        }
        return *this;
    }

    TimingsSection::~TimingsSection()                         = default;
    TimingsSection::TimingsSection(TimingsSection&&) noexcept = default;
    TimingsSection& TimingsSection::operator=(TimingsSection&&
    ) noexcept                                                = default;

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
    Time TimingsSection::calculateElapsedTime() const
    {
        return duration_cast<std::chrono::nanoseconds>(_time->totalTime)
                   .count() *
               ns;
    }

    /**
     * @brief calculates the average loop time in ms
     *
     */
    Time TimingsSection::calculateAverageLoopTime() const
    {
        auto time =
            duration_cast<std::chrono::nanoseconds>(_time->totalTime).count();

        return time * ns / static_cast<double>(_steps);
    }

    /**
     * @brief calculates the loop time in s
     *
     */
    Time TimingsSection::calculateLoopTime() const
    {
        auto time = duration_cast<std::chrono::nanoseconds>(_time->lastStepTime)
                        .count();
        return time * ns;
    }

    /**
     * @brief get the name of the timings section
     *
     * @return std::string
     */
    std::string TimingsSection::getName() const { return _name; }

}   // namespace timings
