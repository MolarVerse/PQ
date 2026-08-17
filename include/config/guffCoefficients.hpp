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

#include "strongTypes.hpp"

namespace constants
{
    // clang-format off
    // values calculated from the original SPC publication
    static constexpr double SPC_LJ_C6_OO  = -625.455653639347; // A^6  kcal mol^-1
    static constexpr double SPC_LJ_C12_OO =  629358.472583307; // A^12 kcal mol^-1
    static constexpr LJParams SPC_LJ_PARAMS_OO{.c6 = SPC_LJ_C6_OO, .c12 = SPC_LJ_C12_OO};

    static constexpr double SPC_E_LJ_C6_OO  =  SPC_LJ_C6_OO;  // A^6  kcal mol^-1
    static constexpr double SPC_E_LJ_C12_OO =  SPC_LJ_C12_OO; // A^12 kcal mol^-1
    static constexpr LJParams SPC_E_LJ_PARAMS_OO{.c6 = SPC_E_LJ_C6_OO, .c12 = SPC_E_LJ_C12_OO};

    static constexpr double SPC_FW_LJ_C6_OO  =  -625.5024676571352; // A^6  kcal mol^-1
    static constexpr double SPC_FW_LJ_C12_OO =   629326.9774051674; // A^12 kcal mol^-1
    static constexpr LJParams SPC_FW_LJ_PARAMS_OO{.c6 = SPC_FW_LJ_C6_OO, .c12 = SPC_FW_LJ_C12_OO};

    static constexpr double QSPC_FW_LJ_C6_OO  = -625.5020652114152; // A^6  kcal mol^-1
    static constexpr double QSPC_FW_LJ_C12_OO =  629326.5724987736; // A^12 kcal mol^-1
    static constexpr LJParams QSPC_FW_LJ_PARAMS_OO{.c6 = QSPC_FW_LJ_C6_OO, .c12 = QSPC_FW_LJ_C12_OO};

    static constexpr double SPC_DC_LJ_C6_OO  = -779.8414665600154; // A^6  kcal mol^-1
    static constexpr double SPC_DC_LJ_C12_OO =  773048.5510230307; // A^12 kcal mol^-1
    static constexpr LJParams SPC_DC_LJ_PARAMS_OO{.c6 = SPC_DC_LJ_C6_OO, .c12 = SPC_DC_LJ_C12_OO};

    static constexpr double H2O_DC_LJ_C6_OO  = -590.6923729027751; // A^6  kcal mol^-1
    static constexpr double H2O_DC_LJ_C12_OO =  615459.8371975797; // A^12 kcal mol^-1
    static constexpr LJParams H2O_DC_LJ_PARAMS_OO{.c6 = H2O_DC_LJ_C6_OO, .c12 = H2O_DC_LJ_C12_OO};

    static constexpr double TIP3P_LJ_C6_OO  = -595.067688427684; // A^6  kcal mol^-1
    static constexpr double TIP3P_LJ_C12_OO =  582015.099443679; // A^12 kcal mol^-1
    static constexpr LJParams TIP3P_LJ_PARAMS_OO{.c6 = TIP3P_LJ_C6_OO, .c12 = TIP3P_LJ_C12_OO};

    static constexpr double OPC3_LJ_C6_OO  = -668.637501352773; // A^6  kcal mol^-1
    static constexpr double OPC3_LJ_C12_OO =  683996.561589584; // A^12 kcal mol^-1
    static constexpr LJParams OPC3_LJ_PARAMS_OO{.c6 = OPC3_LJ_C6_OO, .c12 = OPC3_LJ_C12_OO};

    static constexpr double SPC_MTR_LJ_C6_OO  = -613.527724665392; // A^6  kcal mol^-1
    static constexpr double SPC_MTR_LJ_C12_OO =  629302.103250478; // A^12 kcal mol^-1
    static constexpr LJParams SPC_MTR_LJ_PARAMS_OO{.c6 = SPC_MTR_LJ_C6_OO, .c12 = SPC_MTR_LJ_C12_OO};

    static constexpr double TIP3P_MTR_LJ_C6_OO  = -454.1108986615679; // A^6  kcal mol^-1
    static constexpr double TIP3P_MTR_LJ_C12_OO =  513862.3326959847; // A^12 kcal mol^-1
    static constexpr LJParams TIP3P_MTR_LJ_PARAMS_OO{.c6 = TIP3P_MTR_LJ_C6_OO, .c12 = TIP3P_MTR_LJ_C12_OO};
    // clang-format on

}   // namespace constants

#endif   // _GUFF_COEFFICIENTS_HPP_
