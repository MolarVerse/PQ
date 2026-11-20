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

#include "coulombShiftedPotential.hpp"

#include "constants/internalConversionFactors.hpp"

namespace pot
{
    /**
     * @brief calculate the energy and force of the shifted Coulomb potential
     *
     * @param distance
     * @param chargeProduct
     * @param config Reference to the CoulombPotentialConfig object containing
     * the cutoff distances and precomputed values for the shifted potential
     * calculation
     * @return std::pair<double, double>
     */
    std::pair<double, double> CoulombShiftedPotential::calculate(
        double distance,
        double chargeProduct
    ) const
    {
        const auto prefactor           = chargeProduct * COULOMB_PREFACTOR;
        const auto dInv                = 1 / distance;
        const auto deltaCutOff         = _get_cutoff() - distance;
        const auto forceCutOffIntegral = _get_forceCutoff() * deltaCutOff;

        auto energy = dInv - _get_energyCutoff() - forceCutOffIntegral;
        auto force  = (dInv * dInv) - _get_forceCutoff();

        energy *= prefactor;
        force  *= prefactor;

        return {energy, force};
    }
}   // namespace pot
