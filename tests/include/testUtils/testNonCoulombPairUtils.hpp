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

#ifndef _TEST_MORSE_PAIR_UTILS_
#define _TEST_MORSE_PAIR_UTILS_

#include "buckinghamPair.hpp"
#include "guffPair.hpp"
#include "lennardJonesPair.hpp"
#include "morsePair.hpp"

/**
 * @brief struct TestMorsePairUtils
 *
 */
struct TestMorsePairUtils
{
    static const MorseParams& params(const pot::MorsePair* morsePair);
};

/**
 * @brief struct TestLJPairUtils
 *
 */
struct TestLJPairUtils
{
    static const LJParams& params(const pot::LennardJonesPair* ljPair);
};

/**
 * @brief struct TestBuckinghamPairUtils
 *
 */
struct TestBuckinghamPairUtils
{
    static const BuckinghamParams& params(const pot::BuckinghamPair* buckPair);
};

/**
 * @brief struct TestGuffPairUtils
 *
 */
struct TestGuffPairUtils
{
    static const std::array<double, defaults::NUM_GUFF_COEFFICIENTS>& coeffs(
        const pot::GuffPair* guffPair
    );
};

#endif
