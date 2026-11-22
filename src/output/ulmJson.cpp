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

#include "ulmJson.hpp"

#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace out::ulm
{
    /**
     * @brief JSON number as Python's json module writes and reads it
     *
     * @details shortest text that reads back to the same double; integral
     * values keep a ".0" so Python reads them as float
     *
     * @param value
     * @return std::string
     */
    std::string number(double value)
    {
        if (std::isnan(value))
        {
            return "NaN";
        }

        if (std::isinf(value))
        {
            return value > 0.0 ? "Infinity" : "-Infinity";
        }

        constexpr std::size_t        MAX_LENGTH = 32;
        std::array<char, MAX_LENGTH> buffer{};

        const auto result =
            std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);

        std::string text(buffer.data(), result.ptr);
        if (text.find_first_of(".e") == std::string::npos)
        {
            text += ".0";
        }

        return text;
    }

    /**
     * @brief JSON entry of an array stored at offset in the file
     *
     * @param shape
     * @param dtype numpy dtype name, e.g. float64
     * @param offset byte position of the array data
     * @return std::string
     */
    std::string ndarray(
        std::span<const std::size_t> shape,
        std::string_view             dtype,
        std::int64_t                 offset
    )
    {
        std::string text = R"({"ndarray": [[)";

        for (std::size_t dim = 0; dim < shape.size(); ++dim)
        {
            if (dim > 0)
            {
                text += ", ";
            }

            text += std::to_string(shape[dim]);
        }

        text += R"(], ")";
        text += dtype;
        text += R"(", )";
        text += std::to_string(offset);
        text += "]}";

        return text;
    }

}   // namespace out::ulm
