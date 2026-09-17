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

using namespace pot;

/**
 * @brief Construct a new Guff Pair:: Guff Pair object
 *
 * @param cutOff
 * @param coefficients
 */
GuffPair::GuffPair(
    double                                                     cutOff,
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
    double                                                     cutOff,
    double                                                     energyCutoff,
    double                                                     forceCutoff,
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
std::pair<double, double> GuffPair::calculate(double distance) const
{
    double energy = 0.0;
    double force  = 0.0;

    if (const auto coeff1 = _coefficients.at(0); coeff1 != 0.0)
    {
        const auto power2           = _coefficients.at(1);
        const auto distance_power2  = ::pow(distance, power2);
        energy                     += coeff1 / distance_power2;
        force += power2 * coeff1 / (distance_power2 * distance);
    }
    if (const auto coeff3 = _coefficients.at(2); coeff3 != 0.0)
    {
        const auto power4           = _coefficients.at(3);
        const auto distance_power4  = ::pow(distance, power4);
        energy                     += coeff3 / distance_power4;
        force += power4 * coeff3 / (distance_power4 * distance);
    }

    if (const auto coeff5 = _coefficients.at(4); coeff5 != 0.0)
    {
        const auto power6           = _coefficients.at(5);
        const auto distance_power6  = ::pow(distance, power6);
        energy                     += coeff5 / distance_power6;
        force += power6 * coeff5 / (distance_power6 * distance);
    }
    if (const auto coeff7 = _coefficients.at(6); coeff7 != 0.0)
    {
        const auto power8           = _coefficients.at(7);
        const auto distance_power8  = ::pow(distance, power8);
        energy                     += coeff7 / distance_power8;
        force += power8 * coeff7 / (distance_power8 * distance);
    }

    if (const auto coeff9 = _coefficients.at(8); coeff9 != 0.0)
    {
        const auto cexp10 = _coefficients.at(9);
        const auto rExp11 = _coefficients.at(10);

        const auto helper = ::exp(cexp10 * (distance - rExp11));

        energy += coeff9 / (1 + helper);
        force  += coeff9 * cexp10 * helper / ((1 + helper) * (1 + helper));
    }

    if (const auto c12 = _coefficients.at(11); c12 != 0.0)
    {
        const auto cexp13 = _coefficients.at(12);
        const auto rExp14 = _coefficients.at(13);

        const auto helper = ::exp(cexp13 * (distance - rExp14));

        energy += c12 / (1 + helper);
        force  += c12 * cexp13 * helper / ((1 + helper) * (1 + helper));
    }

    if (const auto c15 = _coefficients.at(14); c15 != 0.0)
    {
        const auto cexp16 = _coefficients.at(15);
        const auto rExp17 = _coefficients.at(16);
        const auto n18    = _coefficients.at(17);

        const auto distance_n18 = ::pow(distance - rExp17, n18);
        const auto helper       = c15 * ::exp(cexp16 * distance_n18);

        energy += helper;
        force  += -cexp16 * n18 * distance_n18 / (distance - rExp17) * helper;
    }

    if (const auto c19 = _coefficients.at(18); c19 != 0.0)
    {
        const auto cexp20 = _coefficients.at(19);
        const auto rExp21 = _coefficients.at(20);
        const auto n22    = _coefficients.at(21);

        const auto distance_n22 = ::pow(distance - rExp21, n22);
        const auto helper       = c19 * ::exp(cexp20 * distance_n22);

        energy += helper;
        force  += -cexp20 * n22 * distance_n22 / (distance - rExp21) * helper;
    }

    energy += -_energyCutOff - _forceCutOff * (_radialCutOff - distance);
    force  += -_forceCutOff;

    return {energy, force};
}
