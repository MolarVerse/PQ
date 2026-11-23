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

#include "units.hpp"

namespace pot
{
    /**
     * @brief calculate the energy and force of the shifted Coulomb potential
     *
     * @param distance
     * @param chargeProduct
     * @return std::pair<double, double>
     */
    std::pair<EnergyPerMol, Force> CoulombShiftedPotential::calculate(
        const Distance&      distance,
        const ChargeSquared& chargeProduct
    ) const
    {
        const auto dInv                = 1.0 / distance;
        const auto deltaCutoff         = _get_cutoff() - distance;
        const auto forceCutoffIntegral = _get_forceCutoff() * deltaCutoff;

        auto energy = dInv - _get_energyCutoff() - forceCutoffIntegral;
        auto force  = (dInv * dInv) - _get_forceCutoff();

        const auto prefactor = chargeProduct * coulomb_prefactor;

        return {energy * prefactor, force * prefactor};
    }
}   // namespace pot
