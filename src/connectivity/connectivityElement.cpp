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

#include "connectivityElement.hpp"

using namespace connectivity;
using namespace simulationBox;

ConnectivityElement::ConnectivityElement(
    const std::vector<Molecule *> &molecules,
    const std::vector<size_t>     &atomIndices
)
    : _molecules(molecules), _atomIndices(atomIndices)
{
}

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

std::vector<Molecule *> ConnectivityElement::getMolecules() const
{
    return _molecules;
}

std::vector<size_t> ConnectivityElement::getAtomIndices() const
{
    return _atomIndices;
}