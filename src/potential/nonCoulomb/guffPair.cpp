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

#include "guffPair.hpp"

#include <cmath>   // for exp, pow

using namespace potential;

/**
 * @brief Construct a new Guff Pair:: Guff Pair object
 *
 * @param cutOff
 * @param coefficients
 */
GuffPair::GuffPair(double cutOff, const std::vector<double> &coefficients)
    : NonCoulombPair(cutOff), _coefficients(coefficients)
{
}

/**
 * @brief Construct a new Guff Pair:: Guff Pair object
 *
 * @param cutOff
 * @param energyCutoff
 * @param forceCutoff
 * @param coefficients
 */
GuffPair::GuffPair(
    double                     cutOff,
    double                     energyCutoff,
    double                     forceCutoff,
    const std::vector<double> &coefficients
)
    : NonCoulombPair(cutOff, energyCutoff, forceCutoff),
      _coefficients(coefficients)
{
}

/**
 * @brief calculates the energy and force of a GuffPair
 *
 * Each contribution is gated on its leading coefficient being non-zero. This
 * skips expensive pow/exp calls for terms that contribute nothing, which is
 * common in sparse .guff parametrizations, while keeping behavior identical
 * for any distance > 0.
 *
 * @param distance
 * @return std::pair<double, double>
 */
std::pair<double, double> GuffPair::calculate(double distance) const
{
    double energy = 0.0;
    double force  = 0.0;

    if (const auto c1 = _coefficients[0]; c1 != 0.0)
    {
        const auto n2           = _coefficients[1];
        const auto distance_n2  = ::pow(distance, n2);
        energy                 += c1 / distance_n2;
        force                  += n2 * c1 / (distance_n2 * distance);
    }
    if (const auto c3 = _coefficients[2]; c3 != 0.0)
    {
        const auto n4           = _coefficients[3];
        const auto distance_n4  = ::pow(distance, n4);
        energy                 += c3 / distance_n4;
        force                  += n4 * c3 / (distance_n4 * distance);
    }

    if (const auto c5 = _coefficients[4]; c5 != 0.0)
    {
        const auto n6           = _coefficients[5];
        const auto distance_n6  = ::pow(distance, n6);
        energy                 += c5 / distance_n6;
        force                  += n6 * c5 / (distance_n6 * distance);
    }
    if (const auto c7 = _coefficients[6]; c7 != 0.0)
    {
        const auto n8           = _coefficients[7];
        const auto distance_n8  = ::pow(distance, n8);
        energy                 += c7 / distance_n8;
        force                  += n8 * c7 / (distance_n8 * distance);
    }

    if (const auto c9 = _coefficients[8]; c9 != 0.0)
    {
        const auto cexp10 = _coefficients[9];
        const auto rExp11 = _coefficients[10];

        const auto helper = ::exp(cexp10 * (distance - rExp11));

        energy += c9 / (1 + helper);
        force  += c9 * cexp10 * helper / ((1 + helper) * (1 + helper));
    }

    if (const auto c12 = _coefficients[11]; c12 != 0.0)
    {
        const auto cexp13 = _coefficients[12];
        const auto rExp14 = _coefficients[13];

        const auto helper = ::exp(cexp13 * (distance - rExp14));

        energy += c12 / (1 + helper);
        force  += c12 * cexp13 * helper / ((1 + helper) * (1 + helper));
    }

    if (const auto c15 = _coefficients[14]; c15 != 0.0)
    {
        const auto cexp16 = _coefficients[15];
        const auto rExp17 = _coefficients[16];
        const auto n18    = _coefficients[17];

        const auto distance_n18 = ::pow(distance - rExp17, n18);
        const auto helper       = c15 * ::exp(cexp16 * distance_n18);

        energy += helper;
        force  += -cexp16 * n18 * distance_n18 / (distance - rExp17) * helper;
    }

    if (const auto c19 = _coefficients[18]; c19 != 0.0)
    {
        const auto cexp20 = _coefficients[19];
        const auto rExp21 = _coefficients[20];
        const auto n22    = _coefficients[21];

        const auto distance_n22 = ::pow(distance - rExp21, n22);
        const auto helper       = c19 * ::exp(cexp20 * distance_n22);

        energy += helper;
        force  += -cexp20 * n22 * distance_n22 / (distance - rExp21) * helper;
    }

    energy += -_energyCutOff - _forceCutOff * (_radialCutOff - distance);
    force  += -_forceCutOff;

    return {energy, force};
}

/**
 * @brief get the coefficients of the GuffPair
 *
 * @return std::vector<double>
 */
std::vector<double> GuffPair::getCoefficients() const { return _coefficients; }
