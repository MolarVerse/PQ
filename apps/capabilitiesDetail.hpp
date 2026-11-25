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

#ifndef APPS__CAPABILITIES_DETAIL_HPP
#define APPS__CAPABILITIES_DETAIL_HPP

#include <initializer_list>
#include <string_view>

#include "jsonOutput.hpp"

namespace cli::detail
{
    /**
     * @brief Represents a parameter in the PQ input schema.
     */
    struct JsonParam
    {
        std::string_view name;
        std::string_view type;
    };

    void writeStringArray(
        cli::JsonWriter                        &json,
        std::string_view                        key,
        std::initializer_list<std::string_view> values
    );

    void beginParameter(
        cli::JsonWriter &json,
        const JsonParam &param,
        std::string_view unit
    );
    void beginParameter(cli::JsonWriter &json, const JsonParam &param);

    void writeBuildCapabilities(cli::JsonWriter &json);
    void writeCliCapabilities(cli::JsonWriter &json);
    void writeExternalQMCapabilities(cli::JsonWriter &json);
    void writeParameters(cli::JsonWriter &json);
    void writeInputCapabilities(cli::JsonWriter &json);
}   // namespace cli::detail

#endif   // APPS__CAPABILITIES_DETAIL_HPP
