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

#ifndef OUTPUT__ULM_JSON_HPP
#define OUTPUT__ULM_JSON_HPP

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

/**
 * @brief JSON pieces of the ULM file format of ASE trajectories
 */
namespace out::ulm
{
    [[nodiscard]] std::string number(double value);
    [[nodiscard]] std::string ndarray(
        std::span<const std::size_t> shape,
        std::string_view             dtype,
        std::int64_t                 offset
    );

}   // namespace out::ulm

#endif   // OUTPUT__ULM_JSON_HPP
