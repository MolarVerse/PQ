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

#ifndef _NON_COULOMB_POTENTIAL_HPP_

#define _NON_COULOMB_POTENTIAL_HPP_

#include <cstddef>   // for size_t
#include <vector>    // for vector

#include "typeAliases.hpp"

namespace potential
{
    /**
     * @enum MixingRule
     *
     * @brief MixingRule is an enumeration for the different mixing rules
     *
     * TODO: implement the different mixing rules
     *
     */
    enum class MixingRule : size_t
    {
        NONE
    };

    /**
     * @class NonCoulombPotential
     *
     * @brief NonCoulombPotential is a base class for guff as well as force
     * field non coulomb potentials
     *
     */
    class NonCoulombPotential
    {
       protected:
        MixingRule _mixingRule = MixingRule::NONE;

       public:
        virtual ~NonCoulombPotential() = default;

        [[nodiscard]] virtual pq::SharedNonCoulPair getNonCoulPair(
            const std::vector<size_t>& indices
        ) = 0;

        [[nodiscard]] MixingRule getMixingRule() const;

        void setMixingRule(const MixingRule mixingRule);
    };

}   // namespace potential

#endif   // _NON_COULOMB_POTENTIAL_HPP_