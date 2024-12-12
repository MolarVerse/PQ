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

#ifndef __BUCKINGHAM_HPP__
#define __BUCKINGHAM_HPP__

#include <vector>   // vector

#include "buckinghamPair.hpp"
#include "typeAliases.hpp"

namespace potential
{
    /**
     * @class Buckingham
     *
     * @brief Buckingham is a class for the Buckingham potential
     *
     * @details the _params vector contains the following parameters:
     * - a
     * - dRho
     * - c6
     * - energyCutOff
     * - forceCutOff
     *
     */
    class Buckingham
    {
       private:
        constexpr static size_t _nParams = 5;
        size_t                  _size;

        std::vector<Real> _cutOffs;
        std::vector<Real> _params;

       public:
        Buckingham();
        explicit Buckingham(cul size);

        void resize(cul size);

        void addPair(const BuckinghamPair& pair, cul index1, cul index2);

        [[nodiscard]] std::vector<Real> copyParams() const;
        [[nodiscard]] std::vector<Real> copyCutOffs() const;

        [[nodiscard]] static size_t getNParams();
        [[nodiscard]] size_t        getSize() const;
    };

    class BuckinghamFF : public Buckingham
    {
    };

    class LenardJonesGuff : public Buckingham
    {
    };

}   // namespace potential

#include "buckingham.inl"

#endif   // __BUCKINGHAM_HPP__