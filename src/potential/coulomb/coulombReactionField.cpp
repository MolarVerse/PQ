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

#include "coulombReactionField.hpp"

#include "constants/internalConversionFactors.hpp"
#include "potentialSettings.hpp"

namespace pot
{

    /**
     * @brief Construct a new Coulomb Reaction Field object
     *
     * @param potentialSettings Reference to the PotentialSettings object used
     * to configure interaction cutoffs and other potential-related parameters
     * @param epsilon
     */
    CoulombReactionField::CoulombReactionField(
        const PotentialSettings& potentialSettings,
        double                   epsilon
    )
        : CoulombPotential{potentialSettings},
          _rfPrefactor{(epsilon - 1) / (2.0 * epsilon + 1)}
    {
    }

    /**
     * @brief calculate the energy and force of the reaction field Coulomb
     * potential
     *
     * @param dist
     * @param chargeProduct
     * @param config Reference to the CoulombPotentialConfig object containing
     * the cutoff distances and precomputed values for the reaction field
     * calculation
     * @return std::pair<double, double>
     */
    std::pair<double, double> CoulombReactionField::_calculateInternal(
        double                        dist,
        double                        chargeProduct,
        const CoulombPotentialConfig& config
    ) const
    {
        const auto coulombPrefactor = chargeProduct * COULOMB_PREFACTOR;
        const auto dInv             = 1.0 / dist;
        const auto deltaCutOff      = config.cutoff - dist;
        const auto rCutEnergy       = config.energyCutoff;
        const auto rCutForce        = config.forceCutoff;
        const auto rfCutOffCubed    = _rfPrefactor * config.invCubeCutoff;

        auto energy  = dInv - (2.0 * rCutEnergy) + (dist * rCutForce);
        energy      += rfCutOffCubed * deltaCutOff * deltaCutOff;

        auto force  = (dInv * dInv) - rCutForce;
        force      += 2.0 * rfCutOffCubed * deltaCutOff;

        energy *= coulombPrefactor;
        force  *= coulombPrefactor;

        return {energy, force};
    }

}   // namespace pot
