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

#ifndef __LENNARD_JONES_HPP__
#define __LENNARD_JONES_HPP__

#include <vector>   // vector

#include "lennardJonesPair.hpp"
#include "typeAliases.hpp"

namespace potential
{
    /**
     * @class LennardJones
     *
     * @brief LennardJones is a class for the Lennard-Jones potential
     *
     * @details the _params vector contains the following parameters:
     * - c6
     * - c12
     * - energyCutOff
     * - forceCutOff
     *
     */
    class LennardJones
    {
       private:
        constexpr static size_t _nParams = 4;
        size_t                  _size;

        std::vector<Real> _cutOffs;
        std::vector<Real> _params;

       public:
        LennardJones();
        explicit LennardJones(cul size);

        void resize(cul size);

        void addPair(const LennardJonesPair& pair, cul index1, cul index2);

        [[nodiscard]] std::vector<Real> copyParams() const;
        [[nodiscard]] std::vector<Real> copyCutOffs() const;

        [[nodiscard]] static size_t getNParams();
        [[nodiscard]] size_t        getSize() const;
    };

    class LennardJonesFF : public LennardJones
    {
    };

    class LenardJonesGuff : public LennardJones
    {
    };

}   // namespace potential

#include "lennardJones.inl"

#endif   // __LENNARD_JONES_HPP__