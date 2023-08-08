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
    _nonCoulombRadiusCutOffs.resize(numberOfMoleculeTypes);
    _coulombCoefficients.resize(numberOfMoleculeTypes);
    _coulombEnergyCutOffs.resize(numberOfMoleculeTypes);
    _coulombForceCutOffs.resize(numberOfMoleculeTypes);
    _nonCoulombEnergyCutOffs.resize(numberOfMoleculeTypes);
    _nonCoulombForceCutOffs.resize(numberOfMoleculeTypes);
}

/**
 * @brief resizes all guff related vectors
 *
 * @param m1
 * @param numberOfMoleculeTypes
 */
void SimulationBox::resizeGuff(c_ul m1, c_ul numberOfMoleculeTypes)
{
    _guffCoefficients[m1].resize(numberOfMoleculeTypes);
    _nonCoulombRadiusCutOffs[m1].resize(numberOfMoleculeTypes);
    _coulombCoefficients[m1].resize(numberOfMoleculeTypes);
    _coulombEnergyCutOffs[m1].resize(numberOfMoleculeTypes);
    _coulombForceCutOffs[m1].resize(numberOfMoleculeTypes);
    _nonCoulombEnergyCutOffs[m1].resize(numberOfMoleculeTypes);
    _nonCoulombForceCutOffs[m1].resize(numberOfMoleculeTypes);
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
    _nonCoulombRadiusCutOffs[m1][m2].resize(numberOfAtoms);
    _coulombCoefficients[m1][m2].resize(numberOfAtoms);
    _coulombEnergyCutOffs[m1][m2].resize(numberOfAtoms);
    _coulombForceCutOffs[m1][m2].resize(numberOfAtoms);
    _nonCoulombEnergyCutOffs[m1][m2].resize(numberOfAtoms);
    _nonCoulombForceCutOffs[m1][m2].resize(numberOfAtoms);
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
    _nonCoulombRadiusCutOffs[m1][m2][a1].resize(numberOfAtoms);
    _coulombCoefficients[m1][m2][a1].resize(numberOfAtoms);
    _coulombEnergyCutOffs[m1][m2][a1].resize(numberOfAtoms);
    _coulombForceCutOffs[m1][m2][a1].resize(numberOfAtoms);
    _nonCoulombEnergyCutOffs[m1][m2][a1].resize(numberOfAtoms);
    _nonCoulombForceCutOffs[m1][m2][a1].resize(numberOfAtoms);
}

/**
 * @brief calculates the number of atoms of all molecules in the simulation box
 */
size_t SimulationBox::getNumberOfAtoms() const
{
    auto accumulateFunc = [](const size_t sum, const Molecule &molecule) { return sum + molecule.getNumberOfAtoms(); };

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

    if (const auto molecule = ranges::find_if(_moleculeTypes, isMoleculeType); molecule != _moleculeTypes.end())
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
pair<Molecule *, size_t> SimulationBox::findMoleculeByAtomIndex(const size_t atomIndex)
{
    size_t sum = 0;

    for (auto &molecule : _molecules)
    {
        sum += molecule.getNumberOfAtoms();

        if (sum >= atomIndex) return make_pair(&molecule, atomIndex - (sum - molecule.getNumberOfAtoms()) - 1);
    }

    throw UserInputException("Atom index " + to_string(atomIndex) + " out of range - total number of atoms: " + to_string(sum));
}

/**
 * @brief make external to internal global vdw types map
 *
 */
void SimulationBox::setupExternalToInternalGlobalVdwTypesMap()
{
    auto fillExternalGlobalVdwTypes = [&externalGlobalVdwTypes = _externalGlobalVdwTypes](auto &molecule)
    {
        externalGlobalVdwTypes.insert(externalGlobalVdwTypes.end(),
                                      molecule.getExternalGlobalVDWTypes().begin(),
                                      molecule.getExternalGlobalVDWTypes().end());
    };

    ranges::for_each(_molecules, fillExternalGlobalVdwTypes);

    ranges::sort(_externalGlobalVdwTypes);
    const auto duplicates = ranges::unique(_externalGlobalVdwTypes);
    _externalGlobalVdwTypes.erase(duplicates.begin(), duplicates.end());

    // c++23 with ranges::views::enumerate
    const size_t size = _externalGlobalVdwTypes.size();
    for (size_t i = 0; i < size; ++i)
        _externalToInternalGlobalVDWTypes.try_emplace(_externalGlobalVdwTypes[i], i);
}

/**
 * @brief calculate degrees of freedom
 *
 */
void SimulationBox::calculateDegreesOfFreedom()
{
    auto accumulateFunc = [](const size_t sum, const Molecule &molecule) { return sum + molecule.getDegreesOfFreedom(); };

    _degreesOfFreedom = accumulate(_molecules.begin(), _molecules.end(), 0UL, accumulateFunc) - 3;
}

/**
 * @brief calculate center of mass of all molecules
 *
 */
void SimulationBox::calculateCenterOfMassMolecules()
{
    ranges::for_each(_molecules, [&box = _box](Molecule &molecule) { molecule.calculateCenterOfMass(box.getBoxDimensions()); });
}

/**
 * @brief checks if the coulomb radius cut off is smaller than half of the minimal box dimension
 *
 * @throw UserInputException if coulomb radius cut off is larger than half of the minimal box dimension
 */
void SimulationBox::checkCoulombRadiusCutOff(ExceptionType exceptionType) const
{
    if (getMinimalBoxDimension() < 2.0 * _coulombRadiusCutOff)
    {
        const auto *message = "Coulomb radius cut off is larger than half of the minimal box dimension";
        if (exceptionType == ExceptionType::MANOSTATEXCEPTION)
            throw ManostatException(message);
        else
            throw UserInputException(message);
    }
}