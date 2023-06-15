#include "simulationBox.hpp"
#include "exceptions.hpp"

using namespace std;
using namespace simulationBox;

size_t SimulationBox::getNumberOfParticles() const
{
    size_t numberOfParticles = 0;

    for (const auto &molecule : _molecules)
        numberOfParticles += molecule.getNumberOfAtoms();

    return numberOfParticles;
}

/**
 * @brief find molecule type my moltype
 *
 * @param moltype
 * @return Molecule
 *
 * @throw RstFileException if molecule type not found
 */
[[nodiscard]] Molecule SimulationBox::findMoleculeType(const size_t moltype) const
{
    for (auto &moleculeType : _moleculeTypes)
    {
        if (moleculeType.getMoltype() == moltype)
            return moleculeType;
    }

    throw RstFileException("Molecule type " + to_string(moltype) + " not found");
}

/**
 * @brief calculate degrees of freedom
 *
 * TODO: maybe -3 Ncom
 *
 */
void SimulationBox::calculateDegreesOfFreedom()
{
    for (const auto &molecule : _molecules)
        _degreesOfFreedom += molecule.getNumberOfAtoms() * 3;
}

/**
 * @brief calculate center of mass of all molecules
 *
 */
void SimulationBox::calculateCenterOfMassMolecules()
{
    for (auto &molecule : _molecules)
        molecule.calculateCenterOfMass(_box.getBoxDimensions());
}