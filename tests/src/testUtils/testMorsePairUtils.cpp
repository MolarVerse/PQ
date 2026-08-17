#include "testMorsePairUtils.hpp"

#include "morsePair.hpp"

/**
 * @brief Get the MorseParams from a MorsePair.
 *
 * @param morsePair pointer to the MorsePair object
 * @return const potential::MorseParams& reference to the MorseParams
 */
const potential::MorseParams& TestMorsePairUtils::params(
    const potential::MorsePair* morsePair
)
{
    return morsePair->_params;
}
