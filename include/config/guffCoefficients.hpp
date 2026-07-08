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

namespace constants
{
    // clang-format off
    static constexpr double _SPC_LJ_C6_OO_  = -625.5024676571352; // A^6  kcal mol^-1
    static constexpr double _SPC_LJ_C12_OO_ =  629326.9774051674; // A^12 kcal mol^-1

    static constexpr double _SPC_E_LJ_C6_OO_  =  _SPC_LJ_C6_OO_;  // A^6  kcal mol^-1
    static constexpr double _SPC_E_LJ_C12_OO_ =  _SPC_LJ_C12_OO_; // A^12 kcal mol^-1

    static constexpr double _SPC_FW_LJ_C6_OO_  =  _SPC_LJ_C6_OO_;  // A^6  kcal mol^-1
    static constexpr double _SPC_FW_LJ_C12_OO_ =  _SPC_LJ_C12_OO_; // A^12 kcal mol^-1

    static constexpr double _QSPC_FW_LJ_C6_OO_  = -625.5020652114152; // A^6  kcal mol^-1
    static constexpr double _QSPC_FW_LJ_C12_OO_ =  629326.5724987736; // A^12 kcal mol^-1

    static constexpr double _SPC_DC_LJ_C6_OO_  = -779.8414665600154; // A^6  kcal mol^-1
    static constexpr double _SPC_DC_LJ_C12_OO_ =  773048.5510230307; // A^12 kcal mol^-1

    static constexpr double _H2O_DC_LJ_C6_OO_  = -590.6923729027751; // A^6  kcal mol^-1
    static constexpr double _H2O_DC_LJ_C12_OO_ =  615459.8371975797; // A^12 kcal mol^-1

    static constexpr double _TIP3P_LJ_C6_OO_  = -595.067688427684; // A^6  kcal mol^-1
    static constexpr double _TIP3P_LJ_C12_OO_ =  582015.099443679; // A^12 kcal mol^-1

    static constexpr double _OPC3_LJ_C6_OO_  = -668.637501352773; // A^6  kcal mol^-1
    static constexpr double _OPC3_LJ_C12_OO_ =  683996.561589584; // A^12 kcal mol^-1

    static constexpr double _SPC_MTR_LJ_C6_OO_  = -613.527724665392; // A^6  kcal mol^-1
    static constexpr double _SPC_MTR_LJ_C12_OO_ =  629302.103250478; // A^12 kcal mol^-1

    static constexpr double _TIP3P_MTR_LJ_C6_OO_  = -454.1108986615679; // A^6  kcal mol^-1
    static constexpr double _TIP3P_MTR_LJ_C12_OO_ =  513862.3326959847; // A^12 kcal mol^-1
    // clang-format on

}   // namespace constants

#endif   // _GUFF_COEFFICIENTS_HPP_
