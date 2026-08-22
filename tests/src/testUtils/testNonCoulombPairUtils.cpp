#include "testNonCoulombPairUtils.hpp"

#include "lennardJonesPair.hpp"
#include "morsePair.hpp"

/**
 * @brief Get the MorseParams from a MorsePair.
 *
 * @param morsePair pointer to the MorsePair object
 * @return const potential::MorseParams& reference to the MorseParams
 */
const MorseParams& TestMorsePairUtils::params(
    const potential::MorsePair* morsePair
)
{
    return morsePair->_params;
}

/**
 * @brief Get the LJParams from a LennardJonesPair.
 *
 * @param ljPair pointer to the LennardJonesPair object
 * @return const LJParams& reference to the LJParams
 */
const LJParams& TestLJPairUtils::params(
    const potential::LennardJonesPair* ljPair
)
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
    const potential::BuckinghamPair* buckPair
)
{
    return buckPair->_params;
}
