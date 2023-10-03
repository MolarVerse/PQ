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

#include "molecule.hpp"

#include "atom.hpp"          // for Atom, frameTools
#include "atomMassMap.hpp"   // for atomMassMap

#include <map>      // for map
#include <string>   // for string, operator<=>

using namespace std;
using namespace frameTools;
using namespace linearAlgebra;

/**
 * @brief calculates the center of mass of the molecule
 *
 * @param box
 */
void Molecule::calculateCenterOfMass(const Vec3D &box)
{
    _molMass = 0.0;

    const auto xyz_0 = _atoms[0]->getPosition();

    for (const auto *atom : _atoms)
    {
        const auto   atomName = atom->getElementType();
        const double mass     = constants::atomMassMap.at(atomName);
        auto         position = atom->getPosition();

        position      -= box * round((position - xyz_0) / box);
        _centerOfMass += mass * position;

        _molMass += mass;
    }

    _centerOfMass /= _molMass;
}