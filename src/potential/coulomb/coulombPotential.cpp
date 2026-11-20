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
        : _cutoff(potentialSettings.get_coulombConfig().cutoff),
          _energyCutoff(potentialSettings.get_coulombConfig().energyCutoff),
          _forceCutoff(potentialSettings.get_coulombConfig().forceCutoff),
          _invCubedCutoff(potentialSettings.get_coulombConfig().invCubeCutoff)
    {
    }

}   // namespace pot
