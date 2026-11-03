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

#ifndef CONFIG__UNITS_HPP
#define CONFIG__UNITS_HPP

/// NOTE FOR DEVELOPERS
/// This header provides convenient type aliases for commonly used angular
/// units. 'rad' represents radians and 'deg' represents degrees. Use this
/// header with caution to avoid ambiguity between radians and degrees and also
/// only in cpp source files. It is recommended to include this header only in
/// implementation files (.cpp) and not in header files to prevent potential
/// conflicts and maintain clarity in unit usage throughout the codebase.

#include <mstd/units.hpp>

// NOLINTBEGIN(readability-identifier-length)

inline constexpr auto rad = mstd::units::rad;
inline constexpr auto deg = mstd::units::deg;

inline constexpr auto ps = mstd::units::ps;
inline constexpr auto fs = mstd::units::fs;

// NOLINTEND(readability-identifier-length)
#endif   // CONFIG__UNITS_HPP
