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

using namespace potential;

/**
 * @brief resizes the outermost vector of the 4d vector _guffNonCoulombPairs
 *
 * @param numberOfMoleculeTypes
 */
void GuffNonCoulomb::resizeGuff(const size_t numberOfMoleculeTypes)
{
    _guffNonCoulombPairs.resize(numberOfMoleculeTypes);
}

/**
 * @brief resizes the second outermost vector of the 4d vector
 * _guffNonCoulombPairs
 *
 * @param m1
 * @param numberOfMoleculeTypes
 */
void GuffNonCoulomb::resizeGuff(
    const size_t m1,
    const size_t numberOfMoleculeTypes
)
{
    _guffNonCoulombPairs[m1].resize(numberOfMoleculeTypes);
}

/**
 * @brief resizes the third outermost vector of the 4d vector
 * _guffNonCoulombPairs
 *
 * @param m1
 * @param m2
 * @param numberOfAtoms
 */
void GuffNonCoulomb::resizeGuff(
    const size_t m1,
    const size_t m2,
    const size_t numberOfAtoms
)
{
    _guffNonCoulombPairs[m1][m2].resize(numberOfAtoms);
}

/**
 * @brief resizes the innermost vector of the 4d vector _guffNonCoulombPairs
 *
 * @param m1
 * @param m2
 * @param a1
 * @param numberOfAtoms
 */
void GuffNonCoulomb::resizeGuff(
    const size_t m1,
    const size_t m2,
    const size_t a1,
    const size_t numberOfAtoms
)
{
    _guffNonCoulombPairs[m1][m2][a1].resize(numberOfAtoms);
}

/***************************
 *                         *
 * standard setter methods *
 *                         *
 ***************************/

/**
 * @brief sets the GuffNonCoulombicPair for the given indices
 *
 * @param indices
 * @param nonCoulombPair
 */
void GuffNonCoulomb::setGuffNonCoulPair(
    const std::vector<size_t>             &indices,
    const std::shared_ptr<NonCoulombPair> &nonCoulombPair
)
{
    const auto m1 = getMolType1(indices) - 1;
    const auto m2 = getMolType2(indices) - 1;
    const auto a1 = getAtomType1(indices);
    const auto a2 = getAtomType2(indices);

    _guffNonCoulombPairs[m1][m2][a1][a2] = nonCoulombPair;
}

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

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
    const auto m1 = getMolType1(indices) - 1;
    const auto m2 = getMolType2(indices) - 1;
    const auto a1 = getAtomType1(indices);
    const auto a2 = getAtomType2(indices);

    return _guffNonCoulombPairs[m1][m2][a1][a2];
}

/**
 * @brief get the 4d vector of shared pointers to NonCoulombPair objects
 *
 * @details it is a 4d vector with the following structure:
 *         _guffNonCoulombPairs[m1][m2][a1][a2]
 *
 *
 * @return pq::SharedNonCoulPairVec4d
 */
pq::SharedNonCoulPairVec4d GuffNonCoulomb::getNonCoulombPairs() const
{
    return _guffNonCoulombPairs;
}

/**
 * @brief get the molecule type 1
 *
 * @param indices
 * @return size_t
 */
size_t GuffNonCoulomb::getMolType1(const std::vector<size_t> &indices) const
{
    return indices[0];
}

/**
 * @brief get the molecule type 2
 *
 * @param indices
 * @return size_t
 */
size_t GuffNonCoulomb::getMolType2(const std::vector<size_t> &indices) const
{
    return indices[1];
}

/**
 * @brief get the atom type 1
 *
 * @param indices
 * @return size_t
 */
size_t GuffNonCoulomb::getAtomType1(const std::vector<size_t> &indices) const
{
    return indices[2];
}

/**
 * @brief get the atom type 2
 *
 * @param indices
 * @return size_t
 */
size_t GuffNonCoulomb::getAtomType2(const std::vector<size_t> &indices) const
{
    return indices[3];
}