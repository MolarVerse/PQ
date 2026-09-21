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

#ifndef _KEY_METADATA_HPP_
#define _KEY_METADATA_HPP_

#include <optional>
#include <string>

namespace input
{
    /**
     * @struct KeyMetadata
     *
     * @brief everything needed to describe a key to a human -- the
     * input-file token itself plus documentation fields
     *
     * @details deliberately separate from parsing behavior (default value,
     * allowed values, customParser, onSet) -- this is what gets shown, not
     * how parsing happens
     *
     */
    struct KeyMetadata
    {
        std::string                name;
        std::string                title;
        std::string                description;
        std::optional<std::string> unit         = std::nullopt;
        std::optional<std::string> errorMessage = std::nullopt;
    };

}   // namespace input
#endif   // _KEY_METADATA_HPP_
