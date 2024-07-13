/*****************************************************************************
<GPL_HEADER>

    PQ
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

#ifndef _GUFF_NON_COULOMB_HPP_

#define _GUFF_NON_COULOMB_HPP_

#include <cstddef>   // size_t
#include <memory>    // shared_ptr
#include <vector>    // vector

#include "nonCoulombPotential.hpp"
#include "typeAliases.hpp"

namespace potential
{
    /**
     * @class GuffNonCoulomb
     *
     * @brief inherits NonCoulombPotential
     *
     */
    class GuffNonCoulomb : public NonCoulombPotential
    {
       private:
        pq::SharedNonCoulPairVec4d _guffNonCoulombPairs;

       public:
        void resizeGuff(const size_t);
        void resizeGuff(const size_t, const size_t);
        void resizeGuff(const size_t, const size_t, const size_t);
        void resizeGuff(const size_t, const size_t, const size_t, const size_t);

        /***************************
         * standard setter methods *
         ***************************/

        void setGuffNonCoulPair(const std::vector<size_t> &, const pq::SharedNonCoulPair &);

        /***************************
         * standard setter methods *
         ***************************/

        [[nodiscard]] pq::SharedNonCoulPair getNonCoulPair(
            const pq::stlVectorUL &indices
        ) override;

        [[nodiscard]] pq::SharedNonCoulPairVec4d getNonCoulombPairs() const;

        [[nodiscard]] size_t getMolType1(const std::vector<size_t> &) const;
        [[nodiscard]] size_t getMolType2(const std::vector<size_t> &) const;
        [[nodiscard]] size_t getAtomType1(const std::vector<size_t> &) const;
        [[nodiscard]] size_t getAtomType2(const std::vector<size_t> &) const;
    };

}   // namespace potential

#endif   // _GUFF_NON_COULOMB_HPP_