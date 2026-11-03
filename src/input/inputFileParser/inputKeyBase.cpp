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

#include "inputKeyBase.hpp"

#include "exceptions.hpp"

namespace input
{
    /**
     * @brief constructs a DeprecatedInputKey with the given key and message
     *
     * @param key the key that is deprecated
     * @param message the deprecation message
     */
    DeprecatedInputKey::DeprecatedInputKey(std::string key, std::string message)
        : _key(std::move(key)), _message(std::move(message))
    {
    }

    /**
     * @brief marks the deprecated key as used and prints the deprecation
     * message
     *
     * @throws exc::InputFileException always, indicating that the deprecated
     * key was used
     */
    void DeprecatedInputKey::deprecated(size_t lineNumber) const
    {
        throw exc::InputFileException(
            "Deprecated key '" + _key + "' used at line " +
            std::to_string(lineNumber) + ".\n" + _message
        );
    }

    /**
     * @brief returns the key of the deprecated input key
     *
     * @return the key of the deprecated input key
     */
    const std::string &DeprecatedInputKey::getKey() const { return _key; }
}   // namespace input
