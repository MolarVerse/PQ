#ifndef _FORCE_FIELD_NON_COULOMB_HPP_

#define _FORCE_FIELD_NON_COULOMB_HPP_

#include "matrix.hpp"
#include "nonCoulombPair.hpp"
#include "nonCoulombPotential.hpp"

namespace potential
{
    class ForceFieldNonCoulomb;
}   // namespace potential

class potential::ForceFieldNonCoulomb : public potential::NonCoulombPotential
{
  public:
    linearAlgebra::Matrix<std::shared_ptr<potential::NonCoulombPair>> _nonCoulombicPairsMatrix;

    [[nodiscard]] std::shared_ptr<NonCoulombPair> getNonCoulombPair(const std::vector<size_t> &molAtomVdwIndices) override
    {
        return _nonCoulombicPairsMatrix[getGlobalVdwType1(molAtomVdwIndices)][getGlobalVdwType2(molAtomVdwIndices)];
    }

    [[nodiscard]] size_t getGlobalVdwType1(const std::vector<size_t> &molAtomVdwIndices) const { return molAtomVdwIndices[4]; }
    [[nodiscard]] size_t getGlobalVdwType2(const std::vector<size_t> &molAtomVdwIndices) const { return molAtomVdwIndices[5]; }
};

#endif   // _FORCE_FIELD_NON_COULOMB_HPP_