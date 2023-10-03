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

#ifndef _CONNECTIVITY_ELEMENT_HPP_

#define _CONNECTIVITY_ELEMENT_HPP_

#include <cstddef>
#include <vector>

namespace simulationBox
{
    class Molecule;   // forward declaration
}

namespace connectivity
{

    /**
     * @class ConnectivityElement
     *
     * @brief Represents a connectivity element between n atoms.
     *
     */
    class ConnectivityElement
    {
      protected:
        std::vector<simulationBox::Molecule *> _molecules;
        std::vector<size_t>                    _atomIndices;

      public:
        ConnectivityElement(const std::vector<simulationBox::Molecule *> &molecules, const std::vector<size_t> &atomIndices)
            : _molecules(molecules), _atomIndices(atomIndices){};

        /***************************
         *                         *
         * standard getter methods *
         *                         *
         ***************************/

        [[nodiscard]] std::vector<simulationBox::Molecule *> getMolecules() const { return _molecules; }
        [[nodiscard]] std::vector<size_t>                    getAtomIndices() const { return _atomIndices; }
    };

}   // namespace connectivity

#endif   // _CONNECTIVITY_ELEMENT_HPP_