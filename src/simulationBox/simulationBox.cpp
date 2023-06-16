#include "simulationBox.hpp"

#include "exceptions.hpp"

using namespace std;
using namespace simulationBox;

/**
 * @brief resizes all guff related vectors
 *
 * @param numberOfMoleculeTypes
 */
void SimulationBox::resizeGuff(c_ul numberOfMoleculeTypes)
{
    _guffCoefficients.resize(numberOfMoleculeTypes);
    _rncCutOffs.resize(numberOfMoleculeTypes);
    _coulombCoefficients.resize(numberOfMoleculeTypes);
    _cEnergyCutOffs.resize(numberOfMoleculeTypes);
    _cForceCutOffs.resize(numberOfMoleculeTypes);
    _ncEnergyCutOffs.resize(numberOfMoleculeTypes);
    _ncForceCutOffs.resize(numberOfMoleculeTypes);
}

/**
 * @brief resizes all guff related vectors
 *
 * @param m1
 * @param numberOfMoleculeTypes
 */
void SimulationBox::resizeGuff(c_ul m1, c_ul numberOfMoleulceTypes)
{
    _guffCoefficients[m1].resize(numberOfMoleulceTypes);
    _rncCutOffs[m1].resize(numberOfMoleulceTypes);
    _coulombCoefficients[m1].resize(numberOfMoleulceTypes);
    _cEnergyCutOffs[m1].resize(numberOfMoleulceTypes);
    _cForceCutOffs[m1].resize(numberOfMoleulceTypes);
    _ncEnergyCutOffs[m1].resize(numberOfMoleulceTypes);
    _ncForceCutOffs[m1].resize(numberOfMoleulceTypes);
}

/**
 * @brief resizes all guff related vectors
 *
 * @param m1
 * @param m2
 * @param numberOfMoleculeTypes
 */
void SimulationBox::resizeGuff(c_ul m1, c_ul m2, c_ul numberOfAtoms)
{
    _guffCoefficients[m1][m2].resize(numberOfAtoms);
    _rncCutOffs[m1][m2].resize(numberOfAtoms);
    _coulombCoefficients[m1][m2].resize(numberOfAtoms);
    _cEnergyCutOffs[m1][m2].resize(numberOfAtoms);
    _cForceCutOffs[m1][m2].resize(numberOfAtoms);
    _ncEnergyCutOffs[m1][m2].resize(numberOfAtoms);
    _ncForceCutOffs[m1][m2].resize(numberOfAtoms);
}

/**
 * @brief resizes all guff related vectors
 *
 * @param m1
 * @param m2
 * @param a1
 * @param numberOfMoleculeTypes
 */
void SimulationBox::resizeGuff(c_ul m1, c_ul m2, c_ul a1, c_ul numberOfAtoms)
{
    _guffCoefficients[m1][m2][a1].resize(numberOfAtoms);
    _rncCutOffs[m1][m2][a1].resize(numberOfAtoms);
    _coulombCoefficients[m1][m2][a1].resize(numberOfAtoms);
    _cEnergyCutOffs[m1][m2][a1].resize(numberOfAtoms);
    _cForceCutOffs[m1][m2][a1].resize(numberOfAtoms);
    _ncEnergyCutOffs[m1][m2][a1].resize(numberOfAtoms);
    _ncForceCutOffs[m1][m2][a1].resize(numberOfAtoms);
}

/**
 * @brief calculates the number of atoms of all molecules in the simulation box
 */
size_t SimulationBox::getNumberOfAtoms() const
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
Molecule SimulationBox::findMoleculeType(const size_t moltype) const
{
    for (auto &moleculeType : _moleculeTypes)
        if (moleculeType.getMoltype() == moltype) return moleculeType;

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