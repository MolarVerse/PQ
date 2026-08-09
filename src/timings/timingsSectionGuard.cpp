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
