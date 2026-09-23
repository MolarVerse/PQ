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

#include "inputConverter.hpp"

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
        using Meta = mstd::enum_meta_t<T>;
        return Meta::from_stringCaseInsensitive(raw);
    }

    /**
     * @brief describes the valid domain of the enum for error messages
     *
     * @return a string listing all allowed enum values
     */
    template <mstd::has_enum_meta T>
    std::string Converter<T>::describeDomain()
    {
        using Meta = mstd::enum_meta_t<T>;

        std::string allowed;
        for (size_t i = 0; i < Meta::size; ++i)
        {
            if (i != 0)
                allowed += ", ";
            allowed += std::string(Meta::names.at(i));
        }

        return allowed;
    }

    /**
     * @brief describes the valid domain of the unsigned integral type for error
     * messages
     *
     * @return a string describing the valid domain
     */
    template <std::unsigned_integral T>
    requires(!std::same_as<T, bool>)
    std::string Converter<T>::describeDomain()
    {
        return "positive integer";
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
    std::string describeDomain()
    {
        if constexpr (mstd::has_enum_meta<T>)
            return Converter<T>::describeDomain();
        else if constexpr (std::same_as<T, bool>)
        {
            std::string options;
            for (const auto& [positive, negative] : boolKeywords)
            {
                if (!options.empty())
                    options += "|";

                options += positive;
                options += "|";
                options += negative;
            }
            return options;
        }
        else if constexpr (std::unsigned_integral<T>)
        {
            return Converter<T>::describeDomain();
        }
        else
        {
            return "<value>";
        }
    }
}   // namespace input

#endif   // _INPUT_CONVERTER_TPP_
