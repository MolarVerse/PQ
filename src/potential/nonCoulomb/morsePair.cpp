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

#include "morsePair.hpp"

#include <cmath>   // for exp

using namespace potential;

/**
 * @brief Construct a new Morse Pair:: Morse Pair object
 *
 * @param vanDerWaalsType1
 * @param vanDerWaalsType2
 * @param cutOff
 * @param params
 */
MorsePair::MorsePair(
    const ExtVdwType   vanDerWaalsType1,
    const ExtVdwType   vanDerWaalsType2,
    const double       cutOff,
    const MorseParams &params
)
    : NonCoulombPair(vanDerWaalsType1, vanDerWaalsType2, cutOff),
      _params(params)
{
}

/**
 * @brief Construct a new Morse Pair:: Morse Pair object
 *
 * @param cutOff
 * @param dissociationEnergy
 * @param wellWidth
 * @param equilibriumDistance
 */
MorsePair::MorsePair(const double cutOff, const MorseParams &params)
    : NonCoulombPair(cutOff), _params(params)
{
}

/**
 * @brief Construct a new Morse Pair:: Morse Pair object
 *
 * @param cutOff
 * @param energyCutoff
 * @param forceCutoff
 * @param params
 */
MorsePair::MorsePair(
    const double       cutOff,
    const double       energyCutoff,
    const double       forceCutoff,
    const MorseParams &params
)
    : NonCoulombPair(cutOff, energyCutoff, forceCutoff), _params(params)
{
}

/**
 * @brief operator overload for the comparison of two MorsePair objects
 *
 * @param other
 * @return true
 * @return false
 */
bool MorsePair::operator==(const MorsePair &other) const
{
    auto isEq = true;

    isEq = isEq && NonCoulombPair::operator==(other);
    isEq = isEq && _params == other._params;

    return isEq;
}

/**
 * @brief calculates the energy and force of a MorsePair
 *
 * @param distance
 * @return std::pair<double, double>
 */
std::pair<double, double> MorsePair::calculate(const double distance) const
{
    const auto deltaEquilibrium = distance - _params.equilibriumDistance;
    const auto expTerm = std::exp(-_params.wellWidth * deltaEquilibrium);
    const auto oneMinusExpTerm = 1.0 - expTerm;

    auto energy =
        _params.dissociationEnergy * oneMinusExpTerm * oneMinusExpTerm;
    energy -= _energyCutOff;
    energy -= _forceCutOff * (_radialCutOff - distance);

    auto force  = -2.0 * _params.dissociationEnergy * _params.wellWidth;
    force      *= expTerm * oneMinusExpTerm;
    force      -= _forceCutOff;

    return {energy, force};
}
