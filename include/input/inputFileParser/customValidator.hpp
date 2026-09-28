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

#ifndef _CUSTOM_VALIDATOR_HPP_
#define _CUSTOM_VALIDATOR_HPP_

#include "keyValidatorBase.hpp"

namespace input
{
    /**
     * Custom validator that uses a user-provided validation function.
     *
     * @tparam T The type of the value to validate.
     */
    template <typename T>
    class CustomValidator : public KeyValidator<T>
    {
       private:
        std::function<bool(const T &)> _validationFunction;
        std::string                    _errorMessage;

       public:
        CustomValidator(
            std::function<bool(const T &)> validationFunction,
            std::string                    errorMessage
        );

        [[nodiscard]]
        bool validate(const T &value) override;

        [[nodiscard]]
        std::string errorMessage() override;
    };
}   // namespace input

#ifndef _CUSTOM_VALIDATOR_TPP_
#include "customValidator.tpp"
#endif

#endif   // _CUSTOM_VALIDATOR_HPP_
