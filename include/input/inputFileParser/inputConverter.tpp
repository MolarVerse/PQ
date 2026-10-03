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

#ifndef _INPUT_CONVERTER_TPP_
#define _INPUT_CONVERTER_TPP_

#include <mstd/string/join.hpp>
#include <set>

#include "inputConverter.hpp"
#include "stringUtilities.hpp"

namespace input
{
    /**
     * @brief attempts to parse an integral value from a raw input-file token
     *
     * @param raw the raw input-file token
     * @return an optional containing the parsed value if successful,
     *         std::nullopt otherwise
     */
    template <std::integral T>
    requires(!std::same_as<T, bool>)
    static std::optional<T> _tryParse(std::string_view raw)
    {
        T          value{};
        const auto result =
            std::from_chars(raw.data(), raw.data() + raw.size(), value);

        if (result.ec != std::errc{} || result.ptr != raw.data() + raw.size())
            return std::nullopt;

        return value;
    }

    /**
     * @brief attempts to parse a signed integral value from a raw input-file
     * token
     *
     * @param raw the raw input-file token
     * @return an optional containing the parsed value if successful,
     *         std::nullopt otherwise
     */
    template <std::signed_integral T>
    requires(!std::same_as<T, bool>)
    std::optional<T> Converter<T>::tryParse(std::string_view raw)
    {
        return _tryParse<T>(raw);
    }

    /**
     * @brief attempts to parse an unsigned integral value from a raw input-file
     * token
     *
     * @param raw the raw input-file token
     * @return an optional containing the parsed value if successful,
     *         std::nullopt otherwise
     */
    template <std::unsigned_integral T>
    requires(!std::same_as<T, bool>)
    std::optional<T> Converter<T>::tryParse(std::string_view raw)
    {
        return _tryParse<T>(raw);
    }

    /**
     * @brief attempts to parse an enum value from a raw input-file token
     *
     * @param raw the raw input-file token
     * @return an optional containing the parsed enum value if successful,
     *         std::nullopt otherwise
     */
    template <mstd::has_enum_meta T>
    std::optional<T> Converter<T>::tryParse(std::string_view raw)
    {
        const auto rawTransformed = utilities::toLowerAndReplaceDashesCopy(raw);
        using Meta                = mstd::enum_meta_t<T>;
        return Meta::from_stringCaseInsensitive(rawTransformed);
    }

    /**
     * @brief describes the valid domain of the enum for error messages
     *
     * @return a string listing all allowed enum values
     */
    template <mstd::has_enum_meta T>
    std::string Converter<T>::describeDomain(const std::vector<T>& notAllowed)
    {
        using Meta = mstd::enum_meta_t<T>;

        std::vector<std::string> allowed;
        for (const auto& spelling : Meta::spellings())
        {
            const auto& name  = spelling.text;
            const auto& value = spelling.value;

            if (std::ranges::find(notAllowed, value) != notAllowed.end())
                continue;

            const auto lowerName = utilities::toLowerCopy(name);

            if (std::ranges::find(allowed, lowerName) != allowed.end())
                continue;

            allowed.push_back(lowerName);
        }

        return "Allowed values: " + mstd::join(allowed, ", ");
    }

    /**
     * @brief describes the valid domain of the unsigned integral type for error
     * messages
     *
     * @return a string describing the valid domain
     */
    template <std::unsigned_integral T>
    requires(!std::same_as<T, bool>)
    std::string Converter<T>::describeDomain(const std::vector<T>& notAllowed)
    {
        std::string message = "Value must be a positive integer";

        for (const auto& value : notAllowed)
            message += ", not allowed: " + std::to_string(value);

        return message;
    }

    /**
     * @brief Constructor for ConverterBase
     *
     * @param key the key associated with the input value
     * @param raw the raw input value as a string
     */
    template <typename T>
    ConverterBase<T>::ConverterBase(std::string key, std::string raw)
        : _key(std::move(key)), _raw(std::move(raw))
    {
    }

    /**
     * @brief describes the valid domain of T for error messages
     *
     * @details falls back to a generic placeholder for types without a
     * describeDomain() (i.e. everything except has_enum_meta<T>)
     *
     * @tparam T
     */
    template <typename T>
    std::string ConverterBase<T>::describeDomain(
        const std::vector<T>& notAllowed
    )
    {
        std::string message = "<value>";
        for (const auto& value : notAllowed)
            message += std::format(", not allowed: {}", value);

        return message;
    }
}   // namespace input

#endif   // _INPUT_CONVERTER_TPP_
