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

#include "trajOutput.hpp"

#include "frame.hpp"      // for Frame
#include "molecule.hpp"   // for Molecule
#include "vector3d.hpp"   // for Vec3D, operator<<

#include <ostream>   // for operator<<, basic_ostream, char_traits, flush
#include <vector>    // for vector

using namespace frameTools;

void TrajOutput::write(Frame &frame)
{
    const auto &molecules = frame.getMolecules();
    const auto &box       = frame.getBox();

    _fp << molecules.size() << " " << box << '\n';
    _fp << '\n' << std::flush;

    for (const auto &molecule : molecules)
    {
        const auto &com = molecule.getCenterOfMass();

        _fp << "COM " << com[0] << " " << com[1] << " " << com[2] << '\n' << std::flush;
    }
}