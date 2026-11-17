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

#include "coulombPotential.hpp"

#include "potentialSettings.hpp"

namespace pot
{
    /**
     * @brief Constructor for CoulombPotential
     *
     * @param potentialSettings Reference to the potential settings object
     */
    CoulombPotential::CoulombPotential(
        const PotentialSettings& potentialSettings
    )
        : _potentialSettings(potentialSettings)
    {
    }

    /**
     * @brief Calculate the energy and force of the Coulomb potential
     *
     * @param distance
     * @param chargeProduct
     * @return std::pair<double, double>
     */
    std::pair<double, double> CoulombPotential::calculate(
        double distance,
        double chargeProduct
    ) const
    {
        return _calculateInternal(
            distance,
            chargeProduct,
            _potentialSettings->get_coulombConfig()
        );
    }

}   // namespace pot
