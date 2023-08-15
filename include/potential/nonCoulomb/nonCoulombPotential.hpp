#ifndef _NON_COULOMB_POTENTIAL_HPP_

#define _NON_COULOMB_POTENTIAL_HPP_

#include "nonCoulombPair.hpp"

#include <memory>
#include <vector>

namespace potential
{
    class NonCoulombPotential;
}   // namespace potential

using c_ul                     = const size_t;
using vector4dNonCoulombicPair = std::vector<std::vector<std::vector<std::vector<std::shared_ptr<potential::NonCoulombPair>>>>>;

/**
 * @class NonCoulombPotential
 *
 * @brief NonCoulombPotential is a base class for guff as well as force field non coulomb potentials
 *
 */
class potential::NonCoulombPotential
{
  public:
    virtual ~NonCoulombPotential() = default;

    [[nodiscard]] virtual std::shared_ptr<NonCoulombPair> getNonCoulombPair(const std::vector<size_t> &molAtomVdwIndices) = 0;
};

#endif   // _NON_COULOMB_POTENTIAL_HPP_