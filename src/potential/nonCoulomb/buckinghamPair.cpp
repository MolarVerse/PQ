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

using namespace potential;

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
    const ExtVdwType        vanDerWaalsType1,
    const ExtVdwType        vanDerWaalsType2,
    const double            cutOff,
    const BuckinghamParams& params
)
    : NonCoulombPair(vanDerWaalsType1, vanDerWaalsType2, cutOff),
      _params(params)
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
    const double            cutOff,
    const BuckinghamParams& params
)
    : NonCoulombPair(cutOff), _params(params)
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
    const double            cutOff,
    const double            energyCutoff,
    const double            forceCutoff,
    const BuckinghamParams& params
)
    : NonCoulombPair(cutOff, energyCutoff, forceCutoff), _params(params)
{
}

/**
 * @brief operator overload for the comparison of two BuckinghamPair objects
 *
 * @param other
 * @return true
 * @return false
 */
bool BuckinghamPair::operator==(const BuckinghamPair& other) const
{
    return NonCoulombPair::operator==(other) && _params == other._params;
}

/**
 * @brief calculates the energy and force of a BuckinghamPair
 *
 * @link https://doi.org/10.1098/rspa.1938.0173
 *
 * @param distance
 * @return std::pair<double, double>
 */
std::pair<double, double> BuckinghamPair::calculate(const double distance) const
{
    const auto distanceThird = distance * distance * distance;
    const auto distanceSixth = distanceThird * distanceThird;
    const auto expTerm       = _params.scaling * ::exp(_params.dRho * distance);

    auto energy  = expTerm + _params.c6 / distanceSixth - _energyCutOff;
    energy      -= _forceCutOff * (_radialCutOff - distance);

    auto force = -_params.dRho * expTerm;

    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
    force += 6.0 * _params.c6 / (distanceSixth * distance) - _forceCutOff;

    return {energy, force};
}
