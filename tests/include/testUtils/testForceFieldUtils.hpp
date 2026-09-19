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

#ifndef _TEST_FORCE_FIELD_UTILS_HPP_
#define _TEST_FORCE_FIELD_UTILS_HPP_

#include "angleForceField.hpp"
#include "bondForceField.hpp"
#include "dihedralForceField.hpp"
#include "jCouplingForceField.hpp"

/**
 * @brief Utility functions for testing force field components
 */
struct TestForceFieldUtils
{
    [[nodiscard]]
    static AngleParams getAngleParams(
        const forceField::AngleForceField &angleForceField
    );

    [[nodiscard]]
    static BondParams getBondParams(
        const forceField::BondForceField &bondForceField
    );

    [[nodiscard]]
    static DihedralParams getDihedralParams(
        const forceField::DihedralForceField &dihedralForceField
    );

    [[nodiscard]]
    static JCouplingParams getJCouplingParams(
        const forceField::JCouplingForceField &jCouplingForceField
    );
};

#endif   // _TEST_FORCE_FIELD_UTILS_HPP_
