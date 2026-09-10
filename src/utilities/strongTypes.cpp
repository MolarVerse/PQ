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

#include "strongTypes.hpp"

#include "mathUtilities.hpp"

/**
 * @brief operator overload for the comparison of two LJParams objects
 *
 * @param other
 * @return true
 * @return false
 */
bool LJParams::operator==(const LJParams &other) const
{
    return utilities::compare(c6, other.c6) &&
           utilities::compare(c12, other.c12);
}

/**
 * @brief compare two MorseParams objects for equality
 *
 * @param other
 * @return true
 * @return false
 */
bool MorseParams::operator==(const MorseParams &other) const
{
    return utilities::compare(dissociationEnergy, other.dissociationEnergy) &&
           utilities::compare(wellWidth, other.wellWidth) &&
           utilities::compare(equilibriumDistance, other.equilibriumDistance);
}

/**
 * @brief compare two BuckinghamParams objects for equality
 *
 * @param other
 * @return true
 * @return false
 */
bool BuckinghamParams::operator==(const BuckinghamParams &other) const
{
    return utilities::compare(scaling, other.scaling) &&
           utilities::compare(dRho, other.dRho) &&
           utilities::compare(c6, other.c6);
}

/**
 * @brief compare two AngleParams objects for equality
 *
 * @param other
 * @return true
 * @return false
 */
bool AngleParams::operator==(const AngleParams &other) const
{
    return utilities::compare(equilibrium, other.equilibrium) &&
           utilities::compare(forceConstant, other.forceConstant);
}

/**
 * @brief compare two BondParams objects for equality
 *
 * @param other
 * @return true
 * @return false
 */
bool BondParams::operator==(const BondParams &other) const
{
    return utilities::compare(equilibrium, other.equilibrium) &&
           utilities::compare(forceConstant, other.forceConstant);
}

/**
 * @brief compare two DihedralParams objects for equality
 *
 * @param other
 * @return true
 * @return false
 */
bool DihedralParams::operator==(const DihedralParams &other) const
{
    return utilities::compare(forceConstant, other.forceConstant) &&
           utilities::compare(frequency, other.frequency) &&
           utilities::compare(phaseShift, other.phaseShift);
}

bool JCouplingParams::operator==(const JCouplingParams &other) const
{
    return utilities::compare(forceConstant, other.forceConstant) &&
           utilities::compare(J0, other.J0) && utilities::compare(a, other.a) &&
           utilities::compare(b, other.b) && utilities::compare(c, other.c) &&
           utilities::compare(phaseShift, other.phaseShift);
}
