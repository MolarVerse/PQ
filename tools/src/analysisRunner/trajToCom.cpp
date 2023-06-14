#include "trajToCom.hpp"
#include "atomMassMap.hpp"

#include <iostream>

using namespace std;
using namespace frameTools;

void TrajToCom::setup()
{
    _configReader = ConfigurationReader(_xyzFiles);
}

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