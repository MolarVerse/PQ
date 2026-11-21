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

#include "coulombWolf.hpp"

#include <cmath>

#include "potentialSettings.hpp"
#include "unit_types.hpp"
#include "units.hpp"

#ifndef M_PI
#define M_PI std::numbers::pi
#endif

namespace pot
{

    /**
     * @brief Construct a new Coulomb Wolf:: Coulomb Wolf object
     *
     * @details this constructor calculates automatically the three need wolf
     * parameters from kappa in order to gain speed
     *
     * @param potentialSettings Reference to the PotentialSettings object used
     * to configure interaction cutoffs and other potential-related parameters
     * @param kappa
     */
    CoulombWolf::CoulombWolf(
        const PotentialSettings& potentialSettings,
        const InvDistance&       kappa
    )
        : CoulombPotential(potentialSettings)
    {
        _kappa            = kappa;
        const auto cutoff = _get_cutoff();
        _wolfParam1       = mstd::erfc(_kappa * cutoff) / cutoff;
        _wolfParam2       = 2.0 * _kappa / ::sqrt(M_PI);

        const auto kappaSquared  = _kappa * _kappa;
        const auto coulRCSquared = cutoff * cutoff;
        const auto expFactor     = mstd::exp(-kappaSquared * coulRCSquared);

        _wolfParam3  = _wolfParam1 / cutoff;
        _wolfParam3 += _wolfParam2 * expFactor / cutoff;
    }

    /**
     * @brief calculate the energy and force of the Coulomb potential with [Wolf
     * summation](https://doi.org/10.1063/1.478738) as long range correction
     *
     * @param distance
     * @param chargeProduct
     * @return std::pair<double, double>
     */
    std::pair<EnergyPerMol, Force> CoulombWolf::calculate(
        const Distance&      distance,
        const ChargeSquared& chargeProduct
    ) const
    {
        const auto kappaDistance = _kappa * distance;
        const auto erfcFactor    = mstd::erfc(kappaDistance);
        const auto expFactor     = mstd::exp(-kappaDistance * kappaDistance);

        auto energy  = (erfcFactor / distance) - _wolfParam1;
        energy      += _wolfParam3 * (distance - _get_cutoff());

        auto force  = erfcFactor / (distance * distance);
        force      += _wolfParam2 * expFactor / distance;
        force      -= _wolfParam3;

        const auto chargeFactor = chargeProduct * coulomb_prefactor;
        return {energy * chargeFactor, force * chargeFactor};
    }

}   // namespace pot
