#include "guffNonCoulomb.hpp"

namespace potential
{
    class NonCoulombPair;
}   // namespace potential

using namespace potential;

/**
 * @brief sets the GuffNonCoulombicPair for the given indices
 *
 * @param indices
 * @param nonCoulombPair
 */
void GuffNonCoulomb::setGuffNonCoulombicPair(const std::vector<size_t>             &indices,
                                             const std::shared_ptr<NonCoulombPair> &nonCoulombPair)
{
    _guffNonCoulombPairs[getMolType1(indices) - 1][getMolType2(indices) - 1][getAtomType1(indices)][getAtomType2(indices)] =
        nonCoulombPair;
}

/**
 * @brief gets a shared pointer to a NonCoulombPair object
 *
 * @param indices
 * @return std::shared_ptr<NonCoulombPair>
 */
std::shared_ptr<NonCoulombPair> GuffNonCoulomb::getNonCoulombPair(const std::vector<size_t> &indices)
{
    return _guffNonCoulombPairs[getMolType1(indices) - 1][getMolType2(indices) - 1][getAtomType1(indices)][getAtomType2(indices)];
}