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

using namespace potential;

/**
 * @brief Construct a new Coulomb Potential:: Coulomb Potential object
 *
 * @details the coulomb energy cutoff is set to 1 / coulombRadiusCutOff and the
 * coulomb force cutoff is set to 1 / (coulombRadiusCutOff *
 * coulombRadiusCutOff) the coulomb pre factor is not included here but later in
 * the calculate function of the derived classes
 *
 * @param coulombRadiusCutOff
 */
CoulombPotential::CoulombPotential(const double coulombRadiusCutOff)
{
    _coulombRadiusCutOff = coulombRadiusCutOff;
    _coulombEnergyCutOff = 1 / _coulombRadiusCutOff;
    _coulombForceCutOff  = 1 / (_coulombRadiusCutOff * _coulombRadiusCutOff);
}

/**
 * @brief sets the coulombRadiusCutOff and calculates the energy and force
 * cutoff - equivalent to the constructor
 *
 * @details coulombPreFactor is not included in the energy and force cutoff
 *
 * @param coulombRadiusCutOff
 */
void CoulombPotential::setCoulombRadiusCutOff(const double coulombRadiusCutOff)
{
    _coulombRadiusCutOff = coulombRadiusCutOff;
    _coulombEnergyCutOff = 1 / _coulombRadiusCutOff;
    _coulombForceCutOff  = 1 / (_coulombRadiusCutOff * _coulombRadiusCutOff);
}