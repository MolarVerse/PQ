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

#ifndef _ENUM_BASE_HPP_
#define _ENUM_BASE_HPP_

#include <array>
#include <string_view>

/**
 * @brief InputAlias template to store input aliases for enum types
 *
 * @tparam T The enum type for which input aliases are defined.
 *
 */
template <typename T>
struct InputAlias
{
    static constexpr std::array<std::pair<std::string_view, T>, 0> value{};
};

#endif   // _ENUM_BASE_HPP_
