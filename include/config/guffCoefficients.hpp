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

#ifndef _GUFF_COEFFICIENTS_HPP_

#define _GUFF_COEFFICIENTS_HPP_

#include <vector>   // for vector

namespace constants
{
    // clang-format off
    static constexpr double _SPC_FW_LJ_C6_OO_  = -625.5024676571352; // A^6 kcal mol^-1
    static constexpr double _SPC_FW_LJ_C12_OO_ =  629326.9774051674; // A^6 kcal mol^-1

    static constexpr double _QSPC_FW_LJ_C6_OO_  = -625.5020652114152; // A^6 kcal mol^-1
    static constexpr double _QSPC_FW_LJ_C12_OO_ =  629326.5724987736; // A^6 kcal mol^-1

    static constexpr double _TIP3P_LJ_C6_OO_  = -595.067688427684; // A^6 kcal mol^-1
    static constexpr double _TIP3P_LJ_C12_OO_ =  582015.099443679; // A^6 kcal mol^-1

    static constexpr double _OPC3_LJ_C6_OO_  = -668.637501352773; // A^6 kcal mol^-1
    static constexpr double _OPC3_LJ_C12_OO_ =  683996.561589584; // A^6 kcal mol^-1
    // clang-format on

}   // namespace constants

#endif   // _GUFF_COEFFICIENTS_HPP_
