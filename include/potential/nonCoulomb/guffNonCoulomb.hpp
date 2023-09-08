#ifndef _GUFF_NON_COULOMB_HPP_

#define _GUFF_NON_COULOMB_HPP_

#include "nonCoulombPotential.hpp"

#include <cstddef>   // size_t
#include <memory>    // shared_ptr
#include <vector>    // vector

namespace potential
{
    class NonCoulombPair;   // forward declaration

    using c_ul                     = const size_t;
    using vector4dNonCoulombicPair = std::vector<std::vector<std::vector<std::vector<std::shared_ptr<NonCoulombPair>>>>>;

    /**
     * @class GuffNonCoulomb
     *
     * @brief inherits NonCoulombPotential
     *
     */
    class GuffNonCoulomb : public NonCoulombPotential
    {
      private:
        vector4dNonCoulombicPair _guffNonCoulombPairs;

      public:
        void resizeGuff(c_ul numberOfMoleculeTypes) { _guffNonCoulombPairs.resize(numberOfMoleculeTypes); }
        void resizeGuff(c_ul m1, c_ul numberOfMoleculeTypes) { _guffNonCoulombPairs[m1].resize(numberOfMoleculeTypes); }
        void resizeGuff(c_ul m1, c_ul m2, c_ul numberOfAtoms) { _guffNonCoulombPairs[m1][m2].resize(numberOfAtoms); }
        void resizeGuff(c_ul m1, c_ul m2, c_ul a1, c_ul numberOfAtoms) { _guffNonCoulombPairs[m1][m2][a1].resize(numberOfAtoms); }

        void setGuffNonCoulombicPair(const std::vector<size_t> &indices, const std::shared_ptr<NonCoulombPair> &nonCoulombPair);

        [[nodiscard]] std::shared_ptr<NonCoulombPair> getNonCoulombPair(const std::vector<size_t> &indices) override;
        [[nodiscard]] vector4dNonCoulombicPair        getNonCoulombPairs() const { return _guffNonCoulombPairs; }

        [[nodiscard]] size_t getMolType1(const std::vector<size_t> &indices) const { return indices[0]; }
        [[nodiscard]] size_t getMolType2(const std::vector<size_t> &indices) const { return indices[1]; }
        [[nodiscard]] size_t getAtomType1(const std::vector<size_t> &indices) const { return indices[2]; }
        [[nodiscard]] size_t getAtomType2(const std::vector<size_t> &indices) const { return indices[3]; }
    };

}   // namespace potential

#endif   // _GUFF_NON_COULOMB_HPP_