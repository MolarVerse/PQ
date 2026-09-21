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

#include "inputConverter.hpp"

#include <unordered_set>

#include "stringUtilities.hpp"

namespace input
{
    /**
     * @brief attempts to parse a double from a raw input-file token
     *
     * @param raw the raw input-file token
     * @return an optional containing the parsed double if successful,
     *         std::nullopt otherwise
     */
    std::optional<double> Converter<double>::tryParse(std::string_view raw)
    {
        double     value{};
        const auto result =
            std::from_chars(raw.data(), raw.data() + raw.size(), value);

        if (result.ec != std::errc{} || result.ptr != raw.data() + raw.size())
            return std::nullopt;

        return value;
    }

    /**
     * @brief attempts to parse a bool from a raw input-file token
     *
     * @param raw the raw input-file token
     * @return an optional containing the parsed bool if successful,
     *         std::nullopt otherwise
     */
    std::optional<bool> Converter<bool>::tryParse(std::string_view raw)
    {
        const auto rawTransformed = utilities::toLowerCopy(raw);

        if (std::unordered_set<std::string_view>{"true", "on", "yes"}.contains(
                rawTransformed
            ))
            return true;
        if (std::unordered_set<std::string_view>{"false", "off", "no"}.contains(
                rawTransformed
            ))
            return false;

        return std::nullopt;
    }

}   // namespace input
