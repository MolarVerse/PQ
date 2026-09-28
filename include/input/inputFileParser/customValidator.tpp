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

#ifndef _CUSTOM_VALIDATOR_TPP_
#define _CUSTOM_VALIDATOR_TPP_

#include "customValidator.hpp"

namespace input
{
    /**
     * Constructor for CustomValidator.
     *
     * @param validationFunction The function used to validate the value.
     * @param errorMessage The error message to throw if validation fails.
     */
    template <typename T>
    CustomValidator<T>::CustomValidator(
        std::function<bool(const T &)> validationFunction,
        std::string                    errorMessage
    )
        : _validationFunction(validationFunction),
          _errorMessage(std::move(errorMessage))
    {
    }

    /**
     * Validates the given value using the custom validation function.
     *
     * @param value The value to validate.
     * @return True if the value is valid, false otherwise.
     */
    template <typename T>
    bool CustomValidator<T>::validate(const T &value)
    {
        return _validationFunction(value);
    }

    /**
     * Returns the error message associated with the custom validator.
     *
     * @return The error message.
     */
    template <typename T>
    std::string CustomValidator<T>::errorMessage()
    {
        return _errorMessage;
    }

}   // namespace input

#endif   // _CUSTOM_VALIDATOR_TPP_
