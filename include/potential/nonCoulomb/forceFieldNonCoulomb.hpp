#ifndef _FORCE_FIELD_NON_COULOMB_HPP_

#define _FORCE_FIELD_NON_COULOMB_HPP_

#include "matrix.hpp"
#include "nonCoulombPotential.hpp"

#include <map>        // for map
#include <memory>     // for shared_ptr
#include <optional>   // for optional
#include <stddef.h>   // for size_t
#include <vector>     // for vector

namespace potential
{
    class NonCoulombPair;   // forward declaration

    class ForceFieldNonCoulomb : public NonCoulombPotential
    {
      private:
        std::vector<std::shared_ptr<NonCoulombPair>>           _nonCoulombicPairsVector;
        linearAlgebra::Matrix<std::shared_ptr<NonCoulombPair>> _nonCoulombicPairsMatrix;

      public:
        void addNonCoulombicPair(const std::shared_ptr<NonCoulombPair> &pair) { _nonCoulombicPairsVector.push_back(pair); }

        void determineInternalGlobalVdwTypes(const std::map<size_t, size_t> &);
        void fillDiagonalElementsOfNonCoulombicPairsMatrix(std::vector<std::shared_ptr<NonCoulombPair>> &);
        void fillNonDiagonalElementsOfNonCoulombicPairsMatrix();
        std::vector<std::shared_ptr<NonCoulombPair>>   getSelfInteractionNonCoulombicPairs() const;
        std::optional<std::shared_ptr<NonCoulombPair>> findNonCoulombicPairByInternalTypes(size_t, size_t) const;

        void initNonCoulombicPairsMatrix(const size_t n)
        {
            _nonCoulombicPairsMatrix = linearAlgebra::Matrix<std::shared_ptr<NonCoulombPair>>(n);
        }

        [[nodiscard]] std::shared_ptr<NonCoulombPair> getNonCoulombPair(const std::vector<size_t> &molAtomVdwIndices) override
        {
            return _nonCoulombicPairsMatrix[getGlobalVdwType1(molAtomVdwIndices)][getGlobalVdwType2(molAtomVdwIndices)];
        }
        [[nodiscard]] size_t getGlobalVdwType1(const std::vector<size_t> &molAtomVdwIndices) const
        {
            return molAtomVdwIndices[4];
        }
        [[nodiscard]] size_t getGlobalVdwType2(const std::vector<size_t> &molAtomVdwIndices) const
        {
            return molAtomVdwIndices[5];
        }
        [[nodiscard]] std::vector<std::shared_ptr<NonCoulombPair>> &getNonCoulombicPairsVector()
        {
            return _nonCoulombicPairsVector;
        }

        [[nodiscard]] linearAlgebra::Matrix<std::shared_ptr<NonCoulombPair>> &getNonCoulombicPairsMatrix()
        {
            return _nonCoulombicPairsMatrix;
        }

        void setNonCoulombicPairsVector(const std::vector<std::shared_ptr<NonCoulombPair>> &nonCoulombicPairsVector)
        {
            _nonCoulombicPairsVector = nonCoulombicPairsVector;
        }
    };

}   // namespace potential

#endif   // _FORCE_FIELD_NON_COULOMB_HPP_