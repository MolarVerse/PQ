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

#ifndef _KEY_VALIDATOR_BASE_HPP_
#define _KEY_VALIDATOR_BASE_HPP_

namespace input
{
    /**
     * @brief Create a shared pointer for the given validator
     *
     * @tparam T The type of the validator
     * @param validator The validator instance
     * @return std::shared_ptr<T> A shared pointer to the validator
     */
    template <typename T>
    std::shared_ptr<T> makeShared(const T &validator)
    {
        return std::make_shared<T>(validator);
    }

    /**
     * @brief Base class for key validators
     *
     * @tparam T The type of the value to be validated
     */
    template <typename T>
    class KeyValidator
    {
       public:
        virtual ~KeyValidator() = default;

        /**
         * @brief Validate the given value
         *
         * @param value The value to be validated
         * @return true if the value is valid, false otherwise
         * @note This function must be implemented by derived classes
         */
        [[nodiscard]]
        virtual bool validate(const T &value) = 0;

        /**
         * @brief Get the error message for the last validation
         *
         * @return std::string The error message
         * @note This function must be implemented by derived classes
         */
        [[nodiscard]]
        virtual std::string errorMessage() = 0;
    };
}   // namespace input

#endif   // _KEY_VALIDATOR_BASE_HPP_
