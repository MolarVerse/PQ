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

#include "inputRegistry.hpp"

#include "parserUtils.hpp"
#include "stringUtilities.hpp"

namespace input
{
    /**
     * @brief Registers a deprecated input key.
     *
     * @param name the name of the deprecated key
     * @return a reference to the registered DeprecatedInputKey
     */
    void InputRegistry::registerDeprecatedKey(
        const DeprecatedInputKey &deprecatedKey
    )
    {
        auto key = std::make_unique<DeprecatedInputKey>(deprecatedKey);
        _deprecatedKeys[deprecatedKey.getKey()] = std::move(key);
    }

    /**
     * @brief Parses a line from the input file and updates the corresponding
     * key's value.
     *
     * @param lineElements the elements of the line, split by whitespace
     * @param lineNumber the line number in the input file
     *
     * @throws exc::InputFileException if the key is unknown or parsing
     * fails
     */
    void InputRegistry::parseLine(
        const std::vector<std::string> &lineElements,
        size_t                          lineNumber
    )
    {
        checkCommand(lineElements, lineNumber);

        const auto key =
            utilities::toLowerAndReplaceDashesCopy(lineElements[0]);
        auto it = _keys.find(key);

        if (it == _keys.end())
        {
            throw exc::InputFileException(
                std::format(
                    "Unknown key \"{}\" at line {} in input file",
                    key,
                    lineNumber
                )
            );
        }

        it->second->parse(lineElements, lineNumber);
    }

    /**
     * @brief Provides a full inventory of every registered key in this
     * registry, e.g. for a startup configuration dump or generated
     * documentation
     *
     * @return std::vector<std::string>
     */
    std::vector<std::string> InputRegistry::describeAll() const
    {
        std::vector<std::string> result;
        result.reserve(_keys.size());

        // we pre-sort here the keys as an unordered map doesn't guarantee
        // order on different platforms
        const auto sortedKeys = [&]()
        {
            std::vector<std::string> keys;
            keys.reserve(_keys.size());
            for (const auto &[name, key] : _keys) keys.push_back(name);
            std::ranges::sort(keys);
            return keys;
        }();

        for (const auto &name : sortedKeys)
            result.push_back(_keys.at(name)->describe());

        return result;
    }

    /**
     * @brief Clears the values of all registered keys in this registry.
     *
     */
    void InputRegistry::clearValues()
    {
        for (auto &[name, key] : _keys) key->clearValue();
    }
}   // namespace input
