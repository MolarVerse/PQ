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

#include "testForceFieldUtils.hpp"

/**
 * @brief Get the angle parameters from the given angle force field.
 *
 * @param angleForceField The angle force field to extract parameters from.
 * @return The angle parameters.
 * @throws std::runtime_error if the angle parameters are not set.
 */
AngleParams TestForceFieldUtils::getAngleParams(
    const forceField::AngleForceField &angleForceField
)
{
    if (!angleForceField._params.has_value())
        throw std::runtime_error("Angle parameters are not set.");

    return *angleForceField._params;
}

/**
 * @brief Get the bond parameters from the given bond force field.
 *
 * @param bondForceField The bond force field to extract parameters from.
 * @return The bond parameters.
 * @throws std::runtime_error if the bond parameters are not set.
 */
BondParams TestForceFieldUtils::getBondParams(
    const forceField::BondForceField &bondForceField
)
{
    if (!bondForceField._params.has_value())
        throw std::runtime_error("Bond parameters are not set.");

    return *bondForceField._params;
}

/**
 * @brief Get the dihedral parameters from the given dihedral force field.
 *
 * @param dihedralForceField The dihedral force field to extract parameters
 * from.
 * @return The dihedral parameters.
 * @throws std::runtime_error if the dihedral parameters are not set.
 */
DihedralParams TestForceFieldUtils::getDihedralParams(
    const forceField::DihedralForceField &dihedralForceField
)
{
    if (!dihedralForceField._params.has_value())
        throw std::runtime_error("Dihedral parameters are not set.");

    return *dihedralForceField._params;
}

/**
 * @brief Get the J-coupling parameters from the given J-coupling force field.
 *
 * @param jCouplingForceField The J-coupling force field to extract parameters
 * from.
 * @return The J-coupling parameters.
 * @throws std::runtime_error if the J-coupling parameters are not set.
 */
JCouplingParams TestForceFieldUtils::getJCouplingParams(
    const forceField::JCouplingForceField &jCouplingForceField
)
{
    if (!jCouplingForceField._params.has_value())
        throw std::runtime_error("J-coupling parameters are not set.");

    return *jCouplingForceField._params;
}
