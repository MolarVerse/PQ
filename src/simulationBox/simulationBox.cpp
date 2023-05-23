#include "simulationBox.hpp"
#include "exceptions.hpp"

using namespace std;

/**
 * @brief find molecule type my moltype
 *
 * @param moltype
 * @return Molecule
 *
 * @throw RstFileException if molecule type not found
 */
Molecule SimulationBox::findMoleculeType(int moltype) const
{
    for (auto &moleculeType : _moleculeTypes)
    {
        if (moleculeType.getMoltype() == moltype)
            return moleculeType;
    }

    throw RstFileException("Molecule type " + to_string(moltype) + " not found");
}

void SimulationBox::calculateDegreesOfFreedom()
{
    for (const auto &molecule : _molecules)
    {
        _degreesOfFreedom += molecule.getNumberOfAtoms() * 3;
    }
}