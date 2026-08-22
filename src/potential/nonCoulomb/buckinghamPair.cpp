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

#include "buckinghamPair.hpp"

#include <cmath>   // for exp

#include "mathUtilities.hpp"   // for compare

using namespace potential;
using namespace utilities;

/**
 * @brief Construct a new Buckingham Pair:: Buckingham Pair object
 *
 * @param vanDerWaalsType1
 * @param vanDerWaalsType2
 * @param cutOff
 * @param a
 * @param dRho
 * @param c6
 */
BuckinghamPair::BuckinghamPair(
    size_t vanDerWaalsType1,
    size_t vanDerWaalsType2,
    double cutOff,
    double scaling,
    double dRho,
    double c6
)
    : NonCoulombPair(vanDerWaalsType1, vanDerWaalsType2, cutOff),
      _a(scaling),
      _dRho(dRho),
      _c6(c6)
{
}

/**
 * @brief Construct a new Buckingham Pair:: Buckingham Pair object
 *
 * @param cutOff
 * @param a
 * @param dRho
 * @param c6
 */
BuckinghamPair::BuckinghamPair(
    double cutOff,
    double scaling,
    double dRho,
    double c6
)
    : NonCoulombPair(cutOff), _a(scaling), _dRho(dRho), _c6(c6)
{
}

/**
 * @brief Construct a new Buckingham Pair:: Buckingham Pair object
 *
 * @param cutOff
 * @param energyCutoff
 * @param forceCutoff
 * @param a
 * @param dRho
 * @param c6
 */
BuckinghamPair::BuckinghamPair(
    double cutOff,
    double energyCutoff,
    double forceCutoff,
    double scaling,
    double dRho,
    double c6
)
    : NonCoulombPair(cutOff, energyCutoff, forceCutoff),
      _a(scaling),
      _dRho(dRho),
      _c6(c6)
{
}

/**
 * @brief operator overload for the comparison of two BuckinghamPair objects
 *
 * @param other
 * @return true
 * @return false
 */
bool BuckinghamPair::operator==(const BuckinghamPair &other) const
{
    auto isEqual = NonCoulombPair::operator==(other);
    isEqual      = isEqual && compare(_a, other._a);
    isEqual      = isEqual && compare(_dRho, other._dRho);
    isEqual      = isEqual && compare(_c6, other._c6);

    return isEqual;
}

/**
 * @brief calculates the energy and force of a BuckinghamPair
 *
 * @link https://doi.org/10.1098/rspa.1938.0173
 *
 * @param distance
 * @return std::pair<double, double>
 */
std::pair<double, double> BuckinghamPair::calculate(double distance) const
{
    const auto distanceThird = distance * distance * distance;
    const auto distanceSixth = distanceThird * distanceThird;
    const auto expTerm       = _a * ::exp(_dRho * distance);

    auto energy  = expTerm + _c6 / distanceSixth - _energyCutOff;
    energy      -= _forceCutOff * (_radialCutOff - distance);

    auto force = -_dRho * expTerm;

    // NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,
    // readability-magic-numbers)
    force += 6.0 * _c6 / (distanceSixth * distance) - _forceCutOff;
    // NOLINTEND(cppcoreguidelines-avoid-magic-numbers,
    // readability-magic-numbers)

    return {energy, force};
}

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

/**
 * @brief get the A parameter
 *
 * @return double
 */
double BuckinghamPair::getA() const { return _a; }

/**
 * @brief get the dRho parameter
 *
 * @return double
 */
double BuckinghamPair::getDRho() const { return _dRho; }

/**
 * @brief get the C6 parameter
 *
 * @return double
 */
double BuckinghamPair::getC6() const { return _c6; }
