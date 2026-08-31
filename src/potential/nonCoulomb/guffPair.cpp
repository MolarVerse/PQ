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
GuffPair::GuffPair(
    const double                                               cutOff,
    const std::array<double, defaults::NUM_GUFF_COEFFICIENTS> &coefficients
)
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
    const double                                               cutOff,
    const double                                               energyCutoff,
    const double                                               forceCutoff,
    const std::array<double, defaults::NUM_GUFF_COEFFICIENTS> &coefficients
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
std::pair<double, double> GuffPair::calculate(const double distance) const
{
    double energy = 0.0;
    double force  = 0.0;

    if (const double c1 = _coefficients.at(0); c1 != 0.0)
    {
        const double n2           = _coefficients.at(1);
        const double distance_n2  = ::pow(distance, n2);
        energy                   += c1 / distance_n2;
        force                    += n2 * c1 / (distance_n2 * distance);
    }
    if (const double c3 = _coefficients.at(2); c3 != 0.0)
    {
        const double n4           = _coefficients.at(3);
        const double distance_n4  = ::pow(distance, n4);
        energy                   += c3 / distance_n4;
        force                    += n4 * c3 / (distance_n4 * distance);
    }

    if (const double c5 = _coefficients.at(4); c5 != 0.0)
    {
        const double n6           = _coefficients.at(5);
        const double distance_n6  = ::pow(distance, n6);
        energy                   += c5 / distance_n6;
        force                    += n6 * c5 / (distance_n6 * distance);
    }
    if (const double c7 = _coefficients.at(6); c7 != 0.0)
    {
        const double n8           = _coefficients.at(7);
        const double distance_n8  = ::pow(distance, n8);
        energy                   += c7 / distance_n8;
        force                    += n8 * c7 / (distance_n8 * distance);
    }

    if (const double c9 = _coefficients.at(8); c9 != 0.0)
    {
        const double cexp10 = _coefficients.at(9);
        const double rExp11 = _coefficients.at(10);

        const double helper = ::exp(cexp10 * (distance - rExp11));

        energy += c9 / (1 + helper);
        force  += c9 * cexp10 * helper / ((1 + helper) * (1 + helper));
    }

    if (const double c12 = _coefficients.at(11); c12 != 0.0)
    {
        const double cexp13 = _coefficients.at(12);
        const double rExp14 = _coefficients.at(13);

        const double helper = ::exp(cexp13 * (distance - rExp14));

        energy += c12 / (1 + helper);
        force  += c12 * cexp13 * helper / ((1 + helper) * (1 + helper));
    }

    if (const double c15 = _coefficients.at(14); c15 != 0.0)
    {
        const double cexp16 = _coefficients.at(15);
        const double rExp17 = _coefficients.at(16);
        const double n18    = _coefficients.at(17);

        const double distance_n18 = ::pow(distance - rExp17, n18);
        const double helper       = c15 * ::exp(cexp16 * distance_n18);

        energy += helper;
        force  += -cexp16 * n18 * distance_n18 / (distance - rExp17) * helper;
    }

    if (const double c19 = _coefficients.at(18); c19 != 0.0)
    {
        const double cexp20 = _coefficients.at(19);
        const double rExp21 = _coefficients.at(20);
        const double n22    = _coefficients.at(21);

        const double distance_n22 = ::pow(distance - rExp21, n22);
        const double helper       = c19 * ::exp(cexp20 * distance_n22);

        energy += helper;
        force  += -cexp20 * n22 * distance_n22 / (distance - rExp21) * helper;
    }

    energy += -_energyCutOff - _forceCutOff * (_radialCutOff - distance);
    force  += -_forceCutOff;

    return {energy, force};
}
