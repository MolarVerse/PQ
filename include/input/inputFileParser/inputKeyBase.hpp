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

#ifndef _INPUT_KEY_BASE_HPP_
#define _INPUT_KEY_BASE_HPP_

namespace input
{
    /**
     * @class InputKeyBase
     *
     * @brief type-erased base for InputKey<T>, so keys of different T can
     * live in the same InputRegistry
     *
     */
    class InputKeyBase
    {
       public:
        virtual ~InputKeyBase() = default;

        /**
         * @brief parses a line from the input file and sets the value of the
         * key
         *
         * @param lineElements the elements of the line from the input file
         * @param lineNumber the line number in the input file
         */
        virtual void parse(
            const std::vector<std::string> &lineElements,
            size_t                          lineNumber
        ) = 0;

        /**
         * @brief returns the name of the input key
         *
         * @return the name of the input key
         */
        [[nodiscard]] virtual const std::string &name() const = 0;

        /**
         * @brief returns whether the input key has been explicitly set
         *
         * @return true if the input key has been set, false otherwise
         */
        [[nodiscard]] virtual bool isSet() const = 0;

        /**
         * @brief human-readable summary: key, title, description, unit,
         * current value, default, and allowed set where applicable
         *
         * @details for docs, startup config dumps, or diagnostics -- not
         * used during parsing
         */
        [[nodiscard]] virtual std::string describe() const = 0;

        /**
         * @brief clears the current value of the input key
         *
         * @details
         *
         * This function resets the current value, effectively marking the key
         * as unset.
         */
        virtual void clearValue() = 0;
    };

    /**
     * @class DeprecatedInputKey
     *
     * @brief placeholder for deprecated input keys
     *
     */
    class DeprecatedInputKey
    {
       private:
        std::string _key;
        std::string _message;

       public:
        explicit DeprecatedInputKey(std::string key, std::string message);

        void deprecated(size_t lineNumber) const;

        [[nodiscard]] const std::string &getKey() const;
    };

}   // namespace input

#endif   // _INPUT_KEY_BASE_HPP_
