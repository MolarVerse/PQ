#ifndef _RANGE_VALIDATOR_TPP_
#define _RANGE_VALIDATOR_TPP_

#include "rangeValidator.hpp"

namespace input
{
    /**
     * @brief Construct a new Range Validator object
     *
     * @param min Minimum value (optional)
     * @param max Maximum value (optional)
     */
    template <typename T>
    RangeValidator<T>::RangeValidator(
        const std::optional<T> &min,
        const std::optional<T> &max
    )
        : _min(min), _max(max)
    {
    }

    /**
     * @brief Validate the given value against the range constraints
     *
     * @param value The value to validate
     * @return true if the value is within the range, false otherwise
     */
    template <typename T>
    bool RangeValidator<T>::validate(const T &value) const
    {
        return (!_min || value >= *_min) && (!_max || value <= *_max);
    }

    /**
     * @brief Get the error message for the range constraints
     *
     * @return std::string The error message
     */
    template <typename T>
    std::string RangeValidator<T>::errorMessage() const
    {
        if (_min && _max)
        {
            return std::format("Value must be between {} and {}", *_min, *_max);
        }

        if (_min)
        {
            return std::format(
                "Value must be greater than or equal to {}",
                *_min
            );
        }

        if (_max)
        {
            return std::format("Value must be less than or equal to {}", *_max);
        }

        return "No range constraints";
    }
}   // namespace input

#endif   // _RANGE_VALIDATOR_TPP_
