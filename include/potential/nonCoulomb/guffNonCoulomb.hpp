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
#include <vector>    // vector

#include "nonCoulombPotential.hpp"

namespace pot
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
        std::vector<std::vector<
            std::vector<std::vector<std::shared_ptr<NonCoulombPair>>>>>
            _guffNonCoulombPairs;

       public:
        void resizeGuff(const size_t);
        void resizeGuff(const size_t, const size_t);
        void resizeGuff(const size_t, const size_t, const size_t);
        void resizeGuff(const size_t, const size_t, const size_t, const size_t);

        /***************************
         * standard setter methods *
         ***************************/

        void setGuffNonCoulPair(
            const std::vector<size_t> &,
            const std::shared_ptr<NonCoulombPair> &
        );

        /***************************
         * standard setter methods *
         ***************************/

        [[nodiscard]]
        std::shared_ptr<NonCoulombPair> getNonCoulPair(
            const std::vector<size_t>         &indices,
            const std::pair<VdwType, VdwType> &vdwTypes
        ) override;

        [[nodiscard]] std::vector<std::vector<
            std::vector<std::vector<std::shared_ptr<NonCoulombPair>>>>>
        getNonCoulombPairs() const;

        [[nodiscard]] size_t getMolType1(const std::vector<size_t> &) const;
        [[nodiscard]] size_t getMolType2(const std::vector<size_t> &) const;
        [[nodiscard]] size_t getAtomType1(const std::vector<size_t> &) const;
        [[nodiscard]] size_t getAtomType2(const std::vector<size_t> &) const;
    };

}   // namespace pot

#endif   // _GUFF_NON_COULOMB_HPP_
