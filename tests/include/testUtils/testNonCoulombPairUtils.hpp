#ifndef _TEST_MORSE_PAIR_UTILS_
#define _TEST_MORSE_PAIR_UTILS_

#include "lennardJonesPair.hpp"
#include "morsePair.hpp"

/**
 * @brief struct TestMorsePairUtils
 *
 */
struct TestMorsePairUtils
{
    static const MorseParams& params(const potential::MorsePair* morsePair);
};

/**
 * @brief struct TestLJPairUtils
 *
 */
struct TestLJPairUtils
{
    static const LJParams& params(const potential::LennardJonesPair* ljPair);
};

#endif
