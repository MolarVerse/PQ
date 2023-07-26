#include "simulationBox.hpp"

#include "exceptions.hpp"

#include <ranges>

using namespace std;
using namespace simulationBox;
using namespace customException;

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
    auto accumulateFunc = [](size_t sum, const Molecule &molecule) { return sum + molecule.getNumberOfAtoms(); };

    return accumulate(_molecules.begin(), _molecules.end(), 0UL, accumulateFunc);
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
    auto isMoleculeType = [moltype](const Molecule &mol) { return mol.getMoltype() == moltype; };

    if (auto molecule = ranges::find_if(_moleculeTypes, isMoleculeType); molecule != _moleculeTypes.end())
        return *molecule;
    else
        throw RstFileException("Molecule type " + to_string(moltype) + " not found");
}

/**
 * @brief find molecule by atom index
 *
 * @details return a pair of a pointer to the molecule and the index of the atom in the molecule
 *
 * @param atomIndex
 * @return pair<Molecule *, size_t>
 */
pair<const Molecule *, size_t> SimulationBox::findMoleculeByAtomIndex(const size_t atomIndex) const
{
    size_t sum = 0;

    for (auto &molecule : _molecules)
    {
        sum += molecule.getNumberOfAtoms();

        if (sum >= atomIndex) return make_pair(&molecule, sum - molecule.getNumberOfAtoms() - 1);
    }

    throw UserInputException("Atom index " + to_string(atomIndex) + " out of range - total number of atoms: " + to_string(sum));
}

/**
 * @brief calculate degrees of freedom
 *
 * TODO: maybe -3 Ncom
 *
 */
void SimulationBox::calculateDegreesOfFreedom()
{
    auto accumulateFunc = [](size_t sum, const Molecule &molecule) { return sum + molecule.getDegreesOfFreedom(); };

    _degreesOfFreedom = accumulate(_molecules.begin(), _molecules.end(), 0UL, accumulateFunc) - 3;
}

/**
 * @brief calculate center of mass of all molecules
 *
 */
void SimulationBox::calculateCenterOfMassMolecules()
{
    ranges::for_each(_molecules, [&_box = _box](Molecule &molecule) { molecule.calculateCenterOfMass(_box.getBoxDimensions()); });
}