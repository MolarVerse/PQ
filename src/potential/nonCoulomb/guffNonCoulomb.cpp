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

#include "guffNonCoulomb.hpp"

namespace potential
{
    class NonCoulombPair;   // forward declaration
}

using namespace potential;

/**
 * @brief sets the GuffNonCoulombicPair for the given indices
 *
 * @param indices
 * @param nonCoulombPair
 */
void GuffNonCoulomb::setGuffNonCoulombicPair(
    const std::vector<size_t>             &indices,
    const std::shared_ptr<NonCoulombPair> &nonCoulombPair
)
{
    _guffNonCoulombPairs[getMolType1(indices) - 1][getMolType2(indices) - 1]
                        [getAtomType1(indices)][getAtomType2(indices)] =
                            nonCoulombPair;
}

/**
 * @brief gets a shared pointer to a NonCoulombPair object
 *
 * @param indices
 * @return std::shared_ptr<NonCoulombPair>
 */
std::shared_ptr<NonCoulombPair> GuffNonCoulomb::getNonCoulPair(
    const std::vector<size_t> &indices
)
{
    return _guffNonCoulombPairs[getMolType1(indices) - 1]
                               [getMolType2(indices) - 1][getAtomType1(indices)]
                               [getAtomType2(indices)];
}