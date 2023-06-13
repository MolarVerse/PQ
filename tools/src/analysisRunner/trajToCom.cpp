#include "trajToCom.hpp"
#include "atomMassMap.hpp"

#include <iostream>

using namespace std;

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

        vector<Vec3D> coms;
        for (size_t i = 0; i < _atomIndices.size(); ++i)
        {
            double molMass = 0.0;
            Vec3D com = {0.0, 0.0, 0.0};

            for (size_t j = 0; j < _numberOfAtomsPerMolecule; ++j)
            {
                const size_t atomIndex = _atomIndices[i];
                const auto atomName = _frame.getElementType(atomIndex);
                const double mass = atomMassMap.at(atomName);

                com += _frame.getPosition(atomIndex) * mass;

                molMass += mass;
                i += 1;
            }
            i -= 1;
            com /= molMass;

            coms.push_back(com);
        }

        cout << coms.size() << endl;
        cout << endl;

        for (const auto &com : coms)
        {
            cout << "COM " << com[0] << " " << com[1] << " " << com[2] << endl;
        }
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
            molecule.addAtom(_frame.getAtom(atomIndex);
            i += 1;
        }
        i -= 1;
    }
}