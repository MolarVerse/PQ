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

#include <ranges>

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

        const auto& keys   = std::views::keys(boolKeywords);
        const auto& values = std::views::values(boolKeywords);

        if (std::ranges::find(keys, rawTransformed) != keys.end())
            return true;
        if (std::ranges::find(values, rawTransformed) != values.end())
            return false;

        return std::nullopt;
    }

    /**
     * @brief attempts to parse a File from a raw input-file token
     *
     * @param raw the raw input-file token
     * @return an optional containing the parsed File if successful,
     *         std::nullopt otherwise
     */
    std::optional<mstd::File> Converter<mstd::File>::tryParse(
        std::string_view raw
    )
    {
        mstd::File file((std::string(raw)));
        if (file.exists())
            return file;

        return std::nullopt;
    }

    /**
     * @brief attempts to parse a std::string from a raw input-file token
     *
     * @param raw the raw input-file token
     * @return an optional containing the parsed std::string if successful,
     *         std::nullopt otherwise
     */
    std::optional<std::string> Converter<std::string>::tryParse(
        std::string_view raw
    )
    {
        return std::string(raw);
    }

    /**
     * @brief describes the domain of valid File inputs
     *
     * @return a string describing the domain
     */
    std::string Converter<mstd::File>::describeDomain()
    {
        return "existing file path";
    }

}   // namespace input
