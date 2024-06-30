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

#include "jCouplingForceField.hpp"

using namespace forceField;
using namespace connectivity;

/**
 * @brief Construct a new JCouplingForceField::JCouplingForceField object
 *
 * @param molecules
 * @param atomIndices
 * @param type
 */
JCouplingForceField::JCouplingForceField(
    const std::vector<pq::Molecule *> &molecules,
    const std::vector<size_t>         &atomIndices,
    const size_t                       type
)
    : Dihedral(molecules, atomIndices), _type(type)
{
}

/***************************
 *                         *
 * standard setter methods *
 *                         *
 ***************************/

/**
 * @brief Set the upper symmetry
 *
 * @param boolean
 */
void JCouplingForceField::setUpperSymmetry(const bool boolean)
{
    _upperSymmetry = boolean;
}

/**
 * @brief Set the lower symmetry
 *
 * @param boolean
 */
void JCouplingForceField::setLowerSymmetry(const bool boolean)
{
    _lowerSymmetry = boolean;
}

/**
 * @brief Set the J0
 *
 * @param J0
 */
void JCouplingForceField::setJ0(const double J0) { _J0 = J0; }

/**
 * @brief Set the force constant
 *
 * @param k
 */
void JCouplingForceField::setForceConstant(const double k)
{
    _forceConstant = k;
}

/**
 * @brief Set the a
 *
 * @param a
 */
void JCouplingForceField::setA(const double a) { _a = a; }

/**
 * @brief Set the b
 *
 * @param b
 */
void JCouplingForceField::setB(const double b) { _b = b; }

/**
 * @brief Set the c
 *
 * @param c
 */
void JCouplingForceField::setC(const double c) { _c = c; }

/**
 * @brief Set the phase shift
 *
 * @param phi
 */
void JCouplingForceField::setPhaseShift(const double phi) { _phaseShift = phi; }

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

/**
 * @brief get the type
 *
 * @return size_t
 */
size_t JCouplingForceField::getType() const { return _type; }

/**
 * @brief get if the upper symmetry is set
 *
 * @return bool
 */
bool JCouplingForceField::getUpperSymmetry() const { return _upperSymmetry; }

/**
 * @brief get if the lower symmetry is set
 *
 * @return bool
 */
bool JCouplingForceField::getLowerSymmetry() const { return _lowerSymmetry; }

/**
 * @brief get the J0
 *
 * @return double
 */
double JCouplingForceField::getJ0() const { return _J0; }

/**
 * @brief get the force constant
 *
 * @return double
 */
double JCouplingForceField::getForceConstant() const { return _forceConstant; }

/**
 * @brief get the a
 *
 * @return double
 */
double JCouplingForceField::getA() const { return _a; }

/**
 * @brief get the b
 *
 * @return double
 */
double JCouplingForceField::getB() const { return _b; }

/**
 * @brief get the c
 *
 * @return double
 */
double JCouplingForceField::getC() const { return _c; }

/**
 * @brief get the phase shift
 *
 * @return double
 */
double JCouplingForceField::getPhaseShift() const { return _phaseShift; }