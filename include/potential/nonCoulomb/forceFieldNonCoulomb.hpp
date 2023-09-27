/*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

#ifndef _FORCE_FIELD_NON_COULOMB_HPP_

#define _FORCE_FIELD_NON_COULOMB_HPP_

#include "matrix.hpp"
#include "nonCoulombPotential.hpp"

#include <algorithm>   // for copy, max
#include <cstddef>     // for size_t
#include <map>         // for map
#include <memory>      // for shared_ptr
#include <optional>    // for optional
#include <vector>      // for vector

namespace potential
{
    class NonCoulombPair;   // forward declaration
}

namespace potential
{
    using c_ul               = const size_t;
    using shared_pair        = std::shared_ptr<NonCoulombPair>;
    using c_shared_pair      = const std::shared_ptr<NonCoulombPair>;
    using vec_shared_pair    = std::vector<std::shared_ptr<NonCoulombPair>>;
    using matrix_shared_pair = linearAlgebra::Matrix<std::shared_ptr<NonCoulombPair>>;

    class ForceFieldNonCoulomb : public NonCoulombPotential
    {
      private:
        vec_shared_pair    _nonCoulombPairsVector;
        matrix_shared_pair _nonCoulombPairsMatrix;

      public:
        void addNonCoulombicPair(c_shared_pair &pair) { _nonCoulombPairsVector.push_back(pair); }

        void setupNonCoulombicCutoffs();
        void determineInternalGlobalVdwTypes(const std::map<size_t, size_t> &);
        void fillDiagonalElementsOfNonCoulombPairsMatrix(vec_shared_pair &diagonalElements);
        void fillOffDiagonalElementsOfNonCoulombPairsMatrix();
        void sortNonCoulombicsPairs(vec_shared_pair &diagonalElements);
        void setOffDiagonalElement(const size_t atomType1, const size_t atomType2);

        [[nodiscard]] vec_shared_pair            getSelfInteractionNonCoulombicPairs() const;
        [[nodiscard]] std::optional<shared_pair> findNonCoulombicPairByInternalTypes(size_t, size_t) const;
        [[nodiscard]] shared_pair                getNonCoulombPair(const std::vector<size_t> &indices) override;

        [[nodiscard]] size_t              getGlobalVdwType1(const std::vector<size_t> &indices) const { return indices[4]; }
        [[nodiscard]] size_t              getGlobalVdwType2(const std::vector<size_t> &indices) const { return indices[5]; }
        [[nodiscard]] vec_shared_pair    &getNonCoulombPairsVector() { return _nonCoulombPairsVector; }
        [[nodiscard]] matrix_shared_pair &getNonCoulombPairsMatrix() { return _nonCoulombPairsMatrix; }

        void setNonCoulombPairsVector(const vec_shared_pair &vec) { _nonCoulombPairsVector = vec; }
        void setNonCoulombPairsMatrix(const matrix_shared_pair &mat) { _nonCoulombPairsMatrix = mat; }
        template <typename T>
        void setNonCoulombPairsMatrix(c_ul index1, c_ul index2, T &value)
        {
            _nonCoulombPairsMatrix[index1][index2] = std::make_shared<T>(value);
        }
    };

}   // namespace potential

#endif   // _FORCE_FIELD_NON_COULOMB_HPP_