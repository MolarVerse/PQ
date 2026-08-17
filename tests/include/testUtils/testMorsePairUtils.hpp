#ifndef _TEST_MORSE_PAIR_UTILS_
#define _TEST_MORSE_PAIR_UTILS_

#include "morsePair.hpp"

/**
 * @brief struct TestMorsePairUtils
 *
 */
struct TestMorsePairUtils
{
    static const potential::MorseParams& params(
        const potential::MorsePair* morsePair
    );
};

#endif
