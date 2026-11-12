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

#ifndef UTILITIES__UNITS_HPP
#define UTILITIES__UNITS_HPP

/// NOTE FOR DEVELOPERS
/// This header provides convenient type aliases for commonly used angular
/// units. 'rad' represents radians and 'deg' represents degrees. Use this
/// header with caution to avoid ambiguity between radians and degrees and also
/// only in cpp source files. It is recommended to include this header only in
/// implementation files (.cpp) and not in header files to prevent potential
/// conflicts and maintain clarity in unit usage throughout the codebase.

#include <mstd/units.hpp>

// NOLINTBEGIN(readability-identifier-length)

inline constexpr auto one = mstd::units::one;

inline constexpr auto rad = mstd::units::rad;
inline constexpr auto deg = mstd::units::deg;

inline constexpr auto fs = mstd::units::fs;
inline constexpr auto ps = mstd::units::ps;
inline constexpr auto s  = mstd::units::s;
inline constexpr auto ms = mstd::units::ms;

inline constexpr auto bar     = mstd::units::bar;
inline constexpr auto Pa      = mstd::units::Pa;
inline constexpr auto per_bar = mstd::units::per_bar;
inline constexpr auto per_Pa  = mstd::units::per_Pa;

inline constexpr auto K = mstd::units::K;

inline constexpr auto per_cm = mstd::units::per_cm;
inline constexpr auto Hz     = mstd::units::Hz;

inline constexpr auto angstrom = mstd::units::angstrom;
inline constexpr auto bohr     = mstd::units::bohr;

inline constexpr auto kcal_per_mol = mstd::units::kcal_per_mol;

// NOLINTEND(readability-identifier-length)
#endif   // UTILITIES__UNITS_HPP
