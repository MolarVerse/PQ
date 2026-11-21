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

#ifndef CONFIG__GUFF_COEFFICIENTS_HPP
#define CONFIG__GUFF_COEFFICIENTS_HPP

#include <mstd/units/new_units.hpp>

#include "strongTypes.hpp"

// values calculated from the original SPC publication
static constexpr C6 SPC_LJ_C6_OO =
    -625.455653639347 * mstd::units::kcal_per_mol *
    (mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom *
     mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom);
static constexpr C12 SPC_LJ_C12_OO =
    629358.472583307 * mstd::units::kcal_per_mol *
    (mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom *
     mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom *
     mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom *
     mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom);
static constexpr LJParams SPC_LJ_PARAMS_OO{
    .c6  = SPC_LJ_C6_OO,
    .c12 = SPC_LJ_C12_OO
};

static constexpr C6       SPC_E_LJ_C6_OO  = SPC_LJ_C6_OO;
static constexpr C12      SPC_E_LJ_C12_OO = SPC_LJ_C12_OO;
static constexpr LJParams SPC_E_LJ_PARAMS_OO{
    .c6  = SPC_E_LJ_C6_OO,
    .c12 = SPC_E_LJ_C12_OO
};

static constexpr C6 SPC_FW_LJ_C6_OO =
    -625.5024676571352 * mstd::units::kcal_per_mol *
    (mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom *
     mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom);
static constexpr C12 SPC_FW_LJ_C12_OO =
    629326.9774051674 * mstd::units::kcal_per_mol *
    (mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom *
     mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom *
     mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom *
     mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom);
static constexpr LJParams SPC_FW_LJ_PARAMS_OO{
    .c6  = SPC_FW_LJ_C6_OO,
    .c12 = SPC_FW_LJ_C12_OO
};

static constexpr C6 QSPC_FW_LJ_C6_OO =
    -625.5020652114152 * mstd::units::kcal_per_mol *
    (mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom *
     mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom);
static constexpr C12 QSPC_FW_LJ_C12_OO =
    629326.5724987736 * mstd::units::kcal_per_mol *
    (mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom *
     mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom *
     mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom *
     mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom);
static constexpr LJParams QSPC_FW_LJ_PARAMS_OO{
    .c6  = QSPC_FW_LJ_C6_OO,
    .c12 = QSPC_FW_LJ_C12_OO
};

static constexpr C6 SPC_DC_LJ_C6_OO =
    -779.8414665600154 * mstd::units::kcal_per_mol *
    (mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom *
     mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom);
static constexpr C12 SPC_DC_LJ_C12_OO =
    773048.5510230307 * mstd::units::kcal_per_mol *
    (mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom *
     mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom *
     mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom *
     mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom);
static constexpr LJParams SPC_DC_LJ_PARAMS_OO{
    .c6  = SPC_DC_LJ_C6_OO,
    .c12 = SPC_DC_LJ_C12_OO
};

static constexpr C6 H2O_DC_LJ_C6_OO =
    -590.6923729027751 * mstd::units::kcal_per_mol *
    (mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom *
     mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom);
static constexpr C12 H2O_DC_LJ_C12_OO =
    615459.8371975797 * mstd::units::kcal_per_mol *
    (mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom *
     mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom *
     mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom *
     mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom);
static constexpr LJParams H2O_DC_LJ_PARAMS_OO{
    .c6  = H2O_DC_LJ_C6_OO,
    .c12 = H2O_DC_LJ_C12_OO
};

static constexpr C6 TIP3P_LJ_C6_OO =
    -595.067688427684 * mstd::units::kcal_per_mol *
    (mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom *
     mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom);
static constexpr C12 TIP3P_LJ_C12_OO =
    582015.099443679 * mstd::units::kcal_per_mol *
    (mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom *
     mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom *
     mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom *
     mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom);
static constexpr LJParams TIP3P_LJ_PARAMS_OO{
    .c6  = TIP3P_LJ_C6_OO,
    .c12 = TIP3P_LJ_C12_OO
};

static constexpr C6 OPC3_LJ_C6_OO =
    -668.637501352773 * mstd::units::kcal_per_mol *
    (mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom *
     mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom);
static constexpr C12 OPC3_LJ_C12_OO =
    683996.561589584 * mstd::units::kcal_per_mol *
    (mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom *
     mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom *
     mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom *
     mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom);
static constexpr LJParams OPC3_LJ_PARAMS_OO{
    .c6  = OPC3_LJ_C6_OO,
    .c12 = OPC3_LJ_C12_OO
};

static constexpr C6 SPC_MTR_LJ_C6_OO =
    -613.527724665392 * mstd::units::kcal_per_mol *
    (mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom *
     mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom);
static constexpr C12 SPC_MTR_LJ_C12_OO =
    629302.103250478 * mstd::units::kcal_per_mol *
    (mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom *
     mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom *
     mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom *
     mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom);
static constexpr LJParams SPC_MTR_LJ_PARAMS_OO{
    .c6  = SPC_MTR_LJ_C6_OO,
    .c12 = SPC_MTR_LJ_C12_OO
};

static constexpr C6 TIP3P_MTR_LJ_C6_OO =
    -454.1108986615679 * mstd::units::kcal_per_mol *
    (mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom *
     mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom);
static constexpr C12 TIP3P_MTR_LJ_C12_OO =
    513862.3326959847 * mstd::units::kcal_per_mol *
    (mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom *
     mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom *
     mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom *
     mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom);
static constexpr LJParams TIP3P_MTR_LJ_PARAMS_OO{
    .c6  = TIP3P_MTR_LJ_C6_OO,
    .c12 = TIP3P_MTR_LJ_C12_OO
};

#endif   // CONFIG__GUFF_COEFFICIENTS_HPP
