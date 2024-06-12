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

#ifndef _M_SHAKE_REFERENCE_HPP_

#define _M_SHAKE_REFERENCE_HPP_

#include <atom.hpp>   // for Atom
#include <cstddef>    // for size_t
#include <memory>     // for unique_ptr
#include <vector>     // for vector

namespace simulationBox
{
    class MoleculeType;   // Forward declaration
}   // namespace simulationBox

namespace constraints
{
    /**
     * @class MShakeReference
     *
     * @brief This class is used to store the reference positions for the mShake
     * algorithm
     *
     */
    class MShakeReference
    {
       private:
        std::shared_ptr<simulationBox::MoleculeType> _moleculeType;
        std::vector<simulationBox::Atom>             _atoms;

       public:
        MShakeReference() = default;

        void storeReferencePositions();

        /***************************
         * standard setter methods *
         ***************************/

        void setMoleculeType(simulationBox::MoleculeType &moltype);
        void setAtoms(const std::vector<simulationBox::Atom> &atoms);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] size_t getNumberOfAtoms() const;
        [[nodiscard]] std::vector<simulationBox::Atom> &getAtoms();
        [[nodiscard]] simulationBox::MoleculeType      &getMoleculeType() const;
    };
}   // namespace constraints

#endif   // _M_SHAKE_REFERENCE_HPP_