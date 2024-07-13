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

#ifndef _POTENTIAL_TPP_

#define _POTENTIAL_TPP_

#include "potential.hpp"

namespace potential
{
    /**
     * @brief make shared pointer of the Coulomb potential
     *
     * @tparam T
     * @param p
     */
    template <typename T>
    void Potential::makeCoulombPotential(T p)
    {
        _coulombPotential = std::make_shared<T>(p);
    }

    /**
     * @brief make shared pointer of the non-Coulomb potential
     *
     * @tparam T
     * @param nonCoulombPotential
     */
    template <typename T>
    void Potential::makeNonCoulombPotential(T nonCoulombPotential)
    {
        _nonCoulombPot = std::make_shared<T>(nonCoulombPotential);
    }

}   // namespace potential

#endif   // _POTENTIAL_TPP_