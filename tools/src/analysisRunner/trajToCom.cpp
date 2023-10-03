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

#include "trajToCom.hpp"

#include "atom.hpp"       // for frameTools
#include "molecule.hpp"   // for Molecule

using namespace std;
using namespace frameTools;

void TrajToCom::setup() { _configReader = ConfigurationReader(_xyzFiles); }

void TrajToCom::run()
{
    while (_configReader.nextFrame())
    {
        _frame = _configReader.getFrame();

        setupMolecules();

        for (auto &molecule : _frame.getMolecules())
        {
            molecule.calculateCenterOfMass(_frame.getBox());
        }

        _trajOutput->write(_frame);
    }
}

void TrajToCom::setupMolecules()
{
    for (size_t i = 0; i < _atomIndices.size(); ++i)
    {
        Molecule molecule(_numberOfAtomsPerMolecule);
        for (size_t j = 0; j < _numberOfAtomsPerMolecule; ++j)
        {
            const size_t atomIndex = _atomIndices[i];
            molecule.addAtom(&(_frame.getAtom(atomIndex)));
            i += 1;
        }
        i -= 1;

        _frame.addMolecule(molecule);
    }
}