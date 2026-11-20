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

#include "mp-units/systems/si/constants.h"

// NOLINTBEGIN(readability-identifier-length)

inline constexpr auto percent = mstd::units::percent;

inline constexpr auto one = mstd::units::one;

inline constexpr auto rad        = mstd::units::rad;
inline constexpr auto deg        = mstd::units::deg;
inline constexpr auto NINETY_DEG = 90 * deg;

inline constexpr auto fs = mstd::units::fs;
inline constexpr auto ps = mstd::units::ps;
inline constexpr auto ns = mstd::units::ns;
inline constexpr auto ms = mstd::units::ms;
inline constexpr auto s  = mstd::units::s;

inline constexpr auto bar     = mstd::units::bar;
inline constexpr auto Pa      = mstd::units::Pa;
inline constexpr auto per_bar = mstd::units::per_bar;
inline constexpr auto per_Pa  = mstd::units::per_Pa;

inline constexpr auto K = mstd::units::K;

inline constexpr auto per_cm = mstd::units::per_cm;
inline constexpr auto Hz     = mstd::units::Hz;

inline constexpr auto angstrom   = mstd::units::angstrom;
inline constexpr auto angstrom2  = angstrom * angstrom;
inline constexpr auto angstrom3  = angstrom * angstrom * angstrom;
inline constexpr auto angstrom6  = angstrom3 * angstrom3;
inline constexpr auto angstrom12 = angstrom6 * angstrom6;
inline constexpr auto bohr       = mstd::units::bohr;
inline constexpr auto bohr3      = bohr * bohr * bohr;
inline constexpr auto litre      = mstd::units::litre;

inline constexpr auto mol = mstd::units::mol;

inline constexpr auto amu = mstd::units::amu;
inline constexpr auto kg  = mstd::units::kg;
inline constexpr auto g   = mstd::units::g;

inline constexpr auto g_per_mol = g / mol;

inline constexpr auto kg_per_litre = kg / litre;

inline constexpr auto kcal_per_mol = mstd::units::kcal_per_mol;
inline constexpr auto hartree      = mstd::units::hartree;
inline constexpr auto eV           = mstd::units::eV;

// constants

inline constexpr auto kb                = mstd::units::boltzmann_constant;
inline constexpr auto avogadro_constant = mstd::units::avogadro_constant;
inline constexpr auto gas_constant      = kb * avogadro_constant;
inline constexpr auto planck            = mstd::units::planck_constant;
inline constexpr auto reduced_planck    = mstd::units::reduced_planck_constant;

inline constexpr auto elementary_charge = mstd::units::elementary_charge;

// 1 / (4 π ε0) per mole, with ε0 = 1 / (μ0 c²)
inline constexpr auto coulomb_prefactor =
    1.0 *
    (mp_units::si::magnetic_constant * mp_units::si::speed_of_light_in_vacuum *
     mp_units::si::speed_of_light_in_vacuum * avogadro_constant) /
    (4.0 * std::numbers::pi);

// NOLINTEND(readability-identifier-length)
#endif   // UTILITIES__UNITS_HPP
