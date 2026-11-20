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

#include <cmath>

#include "unit_types.hpp"
#include "units.hpp"

namespace pot
{
    /**
     * @brief Construct a new Guff Pair:: Guff Pair object
     *
     * @param cutOff
     * @param coefficients
     */
    GuffPair::GuffPair(
        const Pos                                                 &cutOff,
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
        const Pos                                                 &cutOff,
        const EnergyPerMol                                        &energyCutoff,
        const Force                                               &forceCutoff,
        const std::array<double, defaults::NUM_GUFF_COEFFICIENTS> &coefficients
    )
        : NonCoulombPair(cutOff, energyCutoff, forceCutoff),
          _coefficients(coefficients)
    {
    }

    /**
     * @brief calculates the energy and force of a GuffPair
     *
     * Each contribution is gated on its leading coefficient being non-zero.
     * This skips expensive pow/exp calls for terms that contribute nothing,
     * which is common in sparse .guff parametrizations, while keeping behavior
     * identical for any distance > 0.
     *
     * @param distance
     * @return std::pair<EnergyPerMol, Force>
     */
    std::pair<EnergyPerMol, Force> GuffPair::calculate(
        const Pos &distance
    ) const
    {
        EnergyPerMol cumulativeEnergy = EnergyPerMol::zero();
        Force        cumulativeForce  = Force::zero();

        // Power-law term, coefficients [coeff, n] starting at `index`:
        //   E = coeff / r^n          (coeff in kcal/mol * Å^n)
        //   F = -dE/dr = n * E / r
        // Skipped entirely if coeff == 0.
        auto addPowerTerms =
            [this, &distance, &cumulativeEnergy, &cumulativeForce](size_t index)
        {
            const auto coeff = _coefficients.at(index);
            if (coeff == 0.0)
                return;

            const auto power = _coefficients.at(index + 1);

            const auto distance_power =
                mstd::pow(distance.numerical_value_in(Pos::unit), power);

            cumulativeEnergy += coeff / distance_power * kcal_per_mol;
            cumulativeForce +=
                power * (coeff / distance_power * kcal_per_mol) / distance;
        };

        // Fermi-like (logistic) term, coefficients [coeff, cexp, rExp]:
        //   h = exp(cexp * (r - rExp))      (cexp in 1/Å, rExp in Å)
        //   E = coeff / (1 + h)
        //   F = -dE/dr = E * cexp * h / (1 + h)
        // Skipped entirely if coeff == 0.
        auto addExpTerm =
            [this, &distance, &cumulativeEnergy, &cumulativeForce](size_t index)
        {
            const auto coeff = _coefficients.at(index);
            if (coeff == 0.0)
                return;

            const auto cexp = _coefficients.at(index + 1) / angstrom;
            const auto rExp = _coefficients.at(index + 2) * angstrom;

            const auto helper = mstd::exp(cexp * (distance - rExp));

            const auto energy_ = coeff * kcal_per_mol / (1 + helper);

            cumulativeEnergy += energy_;
            cumulativeForce  += energy_ * cexp * helper / (1 + helper);
        };

        // Stretched-exponential term, coefficients [coeff, cexp, rExp, n]:
        //   d = (r - rExp)^n                (rExp in Å, d in Å^n)
        //   E = coeff * exp(cexp * d)       (cexp in 1/Å^n)
        //   F = -dE/dr = -cexp * n * d / (r - rExp) * E
        // Skipped entirely if coeff == 0.
        auto addPowExpTerm =
            [this, &distance, &cumulativeEnergy, &cumulativeForce](size_t index)
        {
            const auto coeff = _coefficients.at(index);
            if (coeff == 0.0)
                return;

            const auto cexp  = _coefficients.at(index + 1);
            const auto rExp  = _coefficients.at(index + 2) * angstrom;
            const auto power = _coefficients.at(index + 3);

            const auto distance_n =
                ::pow((distance - rExp).numerical_value_in(Pos::unit), power);
            const auto helper = coeff * ::exp(cexp * distance_n) * kcal_per_mol;

            cumulativeEnergy += helper;
            cumulativeForce +=
                -cexp * power * distance_n / (distance - rExp) * helper;
        };

        size_t index = 0;

        // the first four terms are power terms
        addPowerTerms(index);
        index += 2;
        addPowerTerms(index);
        index += 2;
        addPowerTerms(index);
        index += 2;
        addPowerTerms(index);
        index += 2;

        // the next two terms are exponential terms
        addExpTerm(index);
        index += 3;
        addExpTerm(index);
        index += 3;

        // the next two terms are power-exponential terms
        addPowExpTerm(index);
        index += 4;
        addPowExpTerm(index);

        cumulativeEnergy += -get_energyCutoff() -
                            get_forceCutoff() * (get_radialCutoff() - distance);
        cumulativeForce += -get_forceCutoff();

        return {cumulativeEnergy, cumulativeForce};
    }

}   // namespace pot
