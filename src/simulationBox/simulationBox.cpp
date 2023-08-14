#include "simulationBox.hpp"

#include "exceptions.hpp"

#include <ranges>

using namespace std;
using namespace simulationBox;
using namespace customException;

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
        throw RstFileException(format("Molecule type {} not found", moltype));
}

/**
 * @brief see if molecule type exists by moltype
 *
 * @param moleculeType
 * @return true
 * @return false
 */
bool SimulationBox::moleculeTypeExists(const size_t moleculeType) const
{
    auto isMoleculeType = [moleculeType](const Molecule &mol) { return mol.getMoltype() == moleculeType; };

    return ranges::find_if(_moleculeTypes, isMoleculeType) != _moleculeTypes.end();
}

/**
 * @brief find molecule type by string
 *
 * @param identifier
 * @return optional<size_t>
 */
optional<size_t> SimulationBox::findMoleculeTypeByString(const string &identifier) const
{
    auto isMoleculeName = [&identifier](const Molecule &mol) { return mol.getName() == identifier; };

    if (const auto molecule = ranges::find_if(_moleculeTypes, isMoleculeName); molecule != _moleculeTypes.end())
        return molecule->getMoltype();
    else
        return nullopt;
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

        if (sum >= atomIndex)
            return make_pair(&molecule, atomIndex - (sum - molecule.getNumberOfAtoms()) - 1);
    }

    throw UserInputException(format("Atom index {} out of range - total number of atoms: {}", atomIndex, sum));
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