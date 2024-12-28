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

#ifndef __MORSE_HPP__
#define __MORSE_HPP__

#include <vector>   // vector

#include "morsePair.hpp"
#include "typeAliases.hpp"

namespace potential
{
    /**
     * @class Morse
     *
     * @brief Morse is a class for the Morse potential
     *
     * @details the _params vector contains the following parameters:
     * - dissociationEnergy
     * - wellWidth
     * - equilibriumDistance
     * - energyCutOff
     * - forceCutOff
     *
     */
    class Morse
    {
       private:
        constexpr static size_t _nParams = 5;
        size_t                  _size;

        std::vector<Real> _cutOffs;
        std::vector<Real> _params;

       public:
        Morse();
        explicit Morse(cul size);

        void resize(cul size);

        void addPair(const MorsePair& pair, cul index1, cul index2);

        [[nodiscard]] std::vector<Real> copyParams() const;
        [[nodiscard]] std::vector<Real> copyCutOffs() const;

        [[nodiscard]] static size_t getNParams();
        [[nodiscard]] size_t        getSize() const;
    };

    class MorseFF : public Morse
    {
    };

    class MorseGuff : public Morse
    {
    };

}   // namespace potential

#include "morse.inl"

#endif   // __MORSE_HPP__