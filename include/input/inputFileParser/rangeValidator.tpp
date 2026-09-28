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
    template <typename T, Greater G, Less L>
    RangeValidator<T, G, L>::RangeValidator(
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
    template <typename T, Greater G, Less L>
    bool RangeValidator<T, G, L>::validate(const T &value)
    {
        if (!_allowInf && std::isinf(static_cast<double>(value)))
        {
            _infCheckFailed = true;
            return false;
        }

        if (!_allowNaN && std::isnan(static_cast<double>(value)))
        {
            _nanCheckFailed = true;
            return false;
        }

        return _isGreater(value) && _isLess(value);
    }

    /**
     * @brief Get the error message for the range constraints
     *
     * @return std::string The error message
     */
    template <typename T, Greater G, Less L>
    std::string RangeValidator<T, G, L>::errorMessage()
    {
        if (_nanCheckFailed)
        {
            _nanCheckFailed = false;
            return "Value must not be NaN";
        }

        if (_infCheckFailed)
        {
            _infCheckFailed = false;
            return "Value must not be infinite";
        }

        if (_min && _max)
            return std::format("Value must be between {} and {}", *_min, *_max);

        if (_min)
            return _isGreaterMsg();

        if (_max)
            return _isLessMsg();

        return "No range constraints";
    }

    /**
     * @brief Check if the value satisfies the greater-than constraint
     *
     * @param value The value to check
     * @return true if the value satisfies the greater-than constraint, false
     * otherwise
     */
    template <typename T, Greater G, Less L>
    constexpr bool RangeValidator<T, G, L>::_isGreater(const T &value) const
    requires(G == Greater::GE)
    {
        return (!_min || value >= *_min);
    }

    /**
     * @brief Check if the value satisfies the less-than constraint
     *
     * @param value The value to check
     * @return true if the value satisfies the less-than constraint, false
     * otherwise
     */
    template <typename T, Greater G, Less L>
    constexpr bool RangeValidator<T, G, L>::_isGreater(const T &value) const
    requires(G == Greater::GT)
    {
        return (!_min || value > *_min);
    }

    /**
     * @brief Check if the value satisfies the less-than constraint
     *
     * @param value The value to check
     * @return true if the value satisfies the less-than constraint, false
     * otherwise
     */
    template <typename T, Greater G, Less L>
    constexpr bool RangeValidator<T, G, L>::_isLess(const T &value) const
    requires(L == Less::LE)
    {
        return (!_max || value <= *_max);
    }

    /**
     * @brief Check if the value satisfies the less-than constraint
     *
     * @param value The value to check
     * @return true if the value satisfies the less-than constraint, false
     * otherwise
     */
    template <typename T, Greater G, Less L>
    constexpr bool RangeValidator<T, G, L>::_isLess(const T &value) const
    requires(L == Less::LT)
    {
        return (!_max || value < *_max);
    }

    /**
     * @brief Get the error message for the greater-than constraint
     *
     * @return std::string The error message
     */
    template <typename T, Greater G, Less L>
    std::string RangeValidator<T, G, L>::_isGreaterMsg() const
    requires(G == Greater::GE)
    {
        return std::format("Value must be greater than or equal to {}", *_min);
    }

    template <typename T, Greater G, Less L>
    std::string RangeValidator<T, G, L>::_isGreaterMsg() const
    requires(G == Greater::GT)
    {
        return std::format("Value must be greater than {}", *_min);
    }

    /**
     * @brief Get the error message for the less-than constraint
     *
     * @return std::string The error message
     */
    template <typename T, Greater G, Less L>
    std::string RangeValidator<T, G, L>::_isLessMsg() const
    requires(L == Less::LE)
    {
        return std::format("Value must be less than or equal to {}", *_max);
    }

    template <typename T, Greater G, Less L>
    std::string RangeValidator<T, G, L>::_isLessMsg() const
    requires(L == Less::LT)
    {
        return std::format("Value must be less than {}", *_max);
    }

}   // namespace input

#endif   // _RANGE_VALIDATOR_TPP_
