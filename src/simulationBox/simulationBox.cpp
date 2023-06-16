#include "simulationBox.hpp"

#include "exceptions.hpp"

using namespace std;
using namespace simulationBox;

void SimulationBox::resizeGuff(c_ul numberOfMoleculeTypes) {
    _guffCoefficients.resize(numberOfMoleculeTypes);
    _rncCutOffs.resize(numberOfMoleculeTypes);
    _coulombCoefficients.resize(numberOfMoleculeTypes);
    _cEnergyCutOffs.resize(numberOfMoleculeTypes);
    _cForceCutOffs.resize(numberOfMoleculeTypes);
    _ncEnergyCutOffs.resize(numberOfMoleculeTypes);
    _ncForceCutOffs.resize(numberOfMoleculeTypes);
}

void SimulationBox::resizeGuff(c_ul m1, c_ul numberOfMoleulceTypes) {
    _guffCoefficients[m1].resize(numberOfMoleulceTypes);
    _rncCutOffs[m1].resize(numberOfMoleulceTypes);
    _coulombCoefficients[m1].resize(numberOfMoleulceTypes);
    _cEnergyCutOffs[m1].resize(numberOfMoleulceTypes);
    _cForceCutOffs[m1].resize(numberOfMoleulceTypes);
    _ncEnergyCutOffs[m1].resize(numberOfMoleulceTypes);
    _ncForceCutOffs[m1].resize(numberOfMoleulceTypes);
}

void SimulationBox::resizeGuff(c_ul m1, c_ul m2, c_ul numberOfAtoms) {
    _guffCoefficients[m1][m2].resize(numberOfAtoms);
    _rncCutOffs[m1][m2].resize(numberOfAtoms);
    _coulombCoefficients[m1][m2].resize(numberOfAtoms);
    _cEnergyCutOffs[m1][m2].resize(numberOfAtoms);
    _cForceCutOffs[m1][m2].resize(numberOfAtoms);
    _ncEnergyCutOffs[m1][m2].resize(numberOfAtoms);
    _ncForceCutOffs[m1][m2].resize(numberOfAtoms);
}

void SimulationBox::resizeGuff(c_ul m1, c_ul m2, c_ul a1, c_ul numberOfAtoms) {
    _guffCoefficients[m1][m2][a1].resize(numberOfAtoms);
    _rncCutOffs[m1][m2][a1].resize(numberOfAtoms);
    _coulombCoefficients[m1][m2][a1].resize(numberOfAtoms);
    _cEnergyCutOffs[m1][m2][a1].resize(numberOfAtoms);
    _cForceCutOffs[m1][m2][a1].resize(numberOfAtoms);
    _ncEnergyCutOffs[m1][m2][a1].resize(numberOfAtoms);
    _ncForceCutOffs[m1][m2][a1].resize(numberOfAtoms);
}

size_t SimulationBox::getNumberOfAtoms() const {
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
[[nodiscard]] Molecule SimulationBox::findMoleculeType(const size_t moltype) const {
    for (auto &moleculeType : _moleculeTypes) {
        if (moleculeType.getMoltype() == moltype) return moleculeType;
    }

    throw RstFileException("Molecule type " + to_string(moltype) + " not found");
}

/**
 * @brief calculate degrees of freedom
 *
 * TODO: maybe -3 Ncom
 *
 */
void SimulationBox::calculateDegreesOfFreedom() {
    for (const auto &molecule : _molecules)
        _degreesOfFreedom += molecule.getNumberOfAtoms() * 3;
}

/**
 * @brief calculate center of mass of all molecules
 *
 */
void SimulationBox::calculateCenterOfMassMolecules() {
    for (auto &molecule : _molecules)
        molecule.calculateCenterOfMass(_box.getBoxDimensions());
}