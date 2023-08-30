#ifndef _NON_COULOMB_POTENTIAL_HPP_

#define _NON_COULOMB_POTENTIAL_HPP_

#include <cstddef>   // for size_t
#include <memory>    // for shared_ptr
#include <vector>    // for vector

namespace potential
{
    class NonCoulombPair;   // forward declaration

    enum class MixingRule : size_t
    {
        NONE
    };

    /**
     * @class NonCoulombPotential
     *
     * @brief NonCoulombPotential is a base class for guff as well as force field non coulomb potentials
     *
     */
    class NonCoulombPotential
    {
      protected:
        MixingRule _mixingRule = MixingRule::NONE;   // no mixing rule TODO: implement (including input file keyword)

      public:
        virtual ~NonCoulombPotential() = default;

        [[nodiscard]] virtual std::shared_ptr<NonCoulombPair> getNonCoulombPair(const std::vector<size_t> &molAtomVdwIndices) = 0;

        [[nodiscard]] MixingRule getMixingRule() const { return _mixingRule; }

        void setMixingRule(const MixingRule mixingRule) { _mixingRule = mixingRule; }
    };

}   // namespace potential

#endif   // _NON_COULOMB_POTENTIAL_HPP_