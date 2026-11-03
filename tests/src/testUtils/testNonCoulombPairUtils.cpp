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

#include "testNonCoulombPairUtils.hpp"

#include "lennardJonesPair.hpp"
#include "morsePair.hpp"

/**
 * @brief Get the MorseParams from a MorsePair.
 *
 * @param morsePair pointer to the MorsePair object
 * @return const pot::MorseParams& reference to the MorseParams
 */
const MorseParams& TestMorsePairUtils::params(const pot::MorsePair* morsePair)
{
    return morsePair->_params;
}

/**
 * @brief Get the LJParams from a LennardJonesPair.
 *
 * @param ljPair pointer to the LennardJonesPair object
 * @return const LJParams& reference to the LJParams
 */
const LJParams& TestLJPairUtils::params(const pot::LennardJonesPair* ljPair)
{
    return ljPair->_params;
}

/**
 * @brief Get the BuckinghamParams from a BuckinghamPair.
 *
 * @param buckPair pointer to the BuckinghamPair object
 * @return const BuckinghamParams& reference to the BuckinghamParams
 */
const BuckinghamParams& TestBuckinghamPairUtils::params(
    const pot::BuckinghamPair* buckPair
)
{
    return buckPair->_params;
}

/**
 * @brief Get the coefficients from a GuffPair.
 *
 * @param guffPair pointer to the GuffPair object
 * @return const std::array<double, defaults::NUM_GUFF_COEFFICIENTS>& reference
 * to the coefficients
 */
const std::array<double, defaults::NUM_GUFF_COEFFICIENTS>& TestGuffPairUtils::
    coeffs(const pot::GuffPair* guffPair)
{
    return guffPair->_coefficients;
}
