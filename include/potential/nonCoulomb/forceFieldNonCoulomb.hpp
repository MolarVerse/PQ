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

#ifndef _FORCE_FIELD_NON_COULOMB_HPP_

#define _FORCE_FIELD_NON_COULOMB_HPP_

#include <cstddef>   // for size_t
#include <map>       // for map

#include "nonCoulombPotential.hpp"

class TestNonCoulombPotentialFF;   // forward declaration

namespace benchSetup
{
    struct BenchNonCoulombFFPot;   // forward declaration
}

namespace potential
{
    class ForceFieldNonCoulomb : public NonCoulombPotential
    {
       private:
        std::vector<std::shared_ptr<NonCoulombPair>> _nonCoulPairsVec;

        struct matrix;
        std::unique_ptr<matrix> _nonCoulPairsMatPtr;

       public:
        ForceFieldNonCoulomb();
        ~ForceFieldNonCoulomb() override;

        ForceFieldNonCoulomb(const ForceFieldNonCoulomb &);
        ForceFieldNonCoulomb &operator=(const ForceFieldNonCoulomb &);
        ForceFieldNonCoulomb(ForceFieldNonCoulomb &&);
        ForceFieldNonCoulomb &operator=(ForceFieldNonCoulomb &&);

        void setupNonCoulombicCutoffs();
        void determineInternalGlobalVdwTypes(const std::map<size_t, size_t> &);
        void fillDiagOfNonCoulPairsMatrix(
            std::vector<std::shared_ptr<NonCoulombPair>> &
        );
        void fillOffDiagOfNonCoulPairsMatrix();
        void sortNonCoulombicsPairs(
            std::vector<std::shared_ptr<NonCoulombPair>> &diagonalElements
        );
        void setOffDiagonalElement(const size_t, const size_t);

        [[nodiscard]]
        std::vector<std::shared_ptr<
            NonCoulombPair>> getSelfInteractionNonCoulPairs() const;

        [[nodiscard]]
        std::
            optional<std::shared_ptr<NonCoulombPair>> findNonCoulPairByInternalTypes(
                const size_t,
                const size_t
            ) const;

        void addNonCoulombicPair(const std::shared_ptr<NonCoulombPair> &pair);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]]
        std::shared_ptr<NonCoulombPair> getNonCoulPair(
            const std::vector<size_t> &indices
        ) override;

        [[nodiscard]]
        size_t getGlobalVdwType1(const std::vector<size_t> &) const;
        [[nodiscard]]
        size_t getGlobalVdwType2(const std::vector<size_t> &) const;

        [[nodiscard]]
        std::vector<std::shared_ptr<NonCoulombPair>> &getNonCoulombPairsVector(
        );

        friend class ::TestNonCoulombPotentialFF;
        friend struct benchSetup::BenchNonCoulombFFPot;

        /***************************
         * standard setter methods *
         ***************************/

        void setNonCoulombPairsVector(
            const std::vector<std::shared_ptr<NonCoulombPair>> &vec
        );
    };

}   // namespace potential

#endif   // _FORCE_FIELD_NON_COULOMB_HPP_
