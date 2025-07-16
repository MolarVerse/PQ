/*****************************************************************************
<GPL_HEADER>

    PQ
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

#include "simulationBox.hpp"

#include <algorithm>   // for sort, unique
#include <format>      // for format
#include <numeric>     // for accumulate
#include <random>      // for random_device, mt19937

#include "constants.hpp"           // for _TEMPERATURE_FACTOR_
#include "exceptions.hpp"          // for RstFileException, UserInputException
#include "potentialSettings.hpp"   // for PotentialSettings
#include "randomNumberGenerator.hpp"   // for randomNumberGenerator
#include "settings.hpp"                // for Settings
#include "stlVector.hpp"               // for rms

using simulationBox::SimulationBox;
using namespace linearAlgebra;
using namespace simulationBox;
using namespace customException;
using namespace constants;
using namespace settings;
using namespace randomNumberGenerator;

/**
 * @brief copy simulationBox object this
 *
 * @details shared_ptrs are not copied but new ones are created
 *
 * @notes copy constructor is not used because it would break semantics here
 *
 * @param toCopy
 */
void SimulationBox::copy(const SimulationBox& toCopy)
{
    *this = toCopy;

    this->_atoms.clear();
    this->_qmAtoms.clear();

    for (size_t i = 0; i < toCopy._atoms.size(); ++i)
    {
        const auto atom = std::make_shared<Atom>(*toCopy._atoms[i]);
        this->_atoms.push_back(atom);
        // TODO: ATTENTION AT THE MOMENT ONLY VALID FOR ALL QM_CALCULATIONS
        //       Probably best would be to remove _qmAtoms at all
        this->_qmAtoms.push_back(atom);
    }

    auto fillAtomsInMolecules = [this](size_t runningIndex, Molecule& molecule)
    {
        const size_t numberOfAtoms = molecule.getNumberOfAtoms();
        molecule.getAtoms().clear();

        for (size_t i = 0; i < numberOfAtoms; ++i)
            molecule.addAtom(this->_atoms[runningIndex++]);

        return runningIndex;
    };

    std::accumulate(
        this->_molecules.begin(),
        this->_molecules.end(),
        0,
        fillAtomsInMolecules
    );
}

/**
 * @brief clone simulationBox object
 *
 * @return std::shared_ptr<SimulationBox>
 */
std::shared_ptr<SimulationBox> SimulationBox::clone() const
{
    return std::make_shared<SimulationBox>(*this);
}

/**
 * @brief finds molecule by moleculeType if (size_t)
 *
 * @param moleculeType
 * @return std::optional<Molecule &>
 */
std::optional<Molecule> SimulationBox::findMolecule(const size_t moleculeType)
{
    auto isMoleculeType = [moleculeType](const Molecule& mol)
    { return mol.getMoltype() == moleculeType; };

    const auto molecule = std::ranges::find_if(_molecules, isMoleculeType);

    if (molecule != _molecules.end())
        return *molecule;
    else
        return std::nullopt;
}

/**
 * @brief adds all atomIndices to _qmCenterAtoms vector
 *
 * @param atomIndices
 *
 * @throw UserInputException if atom index out of range
 */
void SimulationBox::addQMCenterAtoms(const std::vector<int>& atomIndices)
{
    for (const auto index : atomIndices)
    {
        if (index < 0 || (size_t) index >= _atoms.size())
            throw UserInputException(
                std::format("QM center atom index {} out of range", index)
            );

        _qmCenterAtoms.push_back(_atoms[(size_t) index]);
    }
}

/**
 * @brief assigns isQMOnly to all atoms which are in the atomIndices vector
 *
 * @details If an atom is not already in the _qmAtoms vector it is added to it
 *
 * @param atomIndices
 *
 * @throw UserInputException if atom index out of range
 */
void SimulationBox::setupQMOnlyAtoms(const std::vector<int>& atomIndices)
{
    for (const auto index : atomIndices)
    {
        if (index < 0 || (size_t) index >= _atoms.size())
            throw UserInputException(
                std::format("QM only atom index {} out of range", index)
            );

        _atoms[(size_t) index]->setQMOnly(true);

        auto it = std::ranges::find(_qmAtoms, _atoms[(size_t) index]);

        if (it == _qmAtoms.end())
            _qmAtoms.push_back(_atoms[(size_t) index]);
    }
}

/**
 * @brief assigns isMMOnly to all atoms which are in the atomIndices vector
 *
 * @param atomIndices
 *
 * @throw UserInputException if atom index out of range
 * @throw UserInputException if atom is already in QM only list
 */
void SimulationBox::setupMMOnlyAtoms(const std::vector<int>& atomIndices)
{
    for (const auto index : atomIndices)
    {
        if (index < 0 || (size_t) index >= _atoms.size())
            throw UserInputException(
                std::format("MM only atom index {} out of range", index)
            );

        _atoms[(size_t) index]->setMMOnly(true);

        auto it = std::ranges::find(_qmAtoms, _atoms[(size_t) index]);

        if (it != _qmAtoms.end())
            throw UserInputException(std::format(
                "Ambiguous atom index {} - atom is already in QM only list "
                "- cannot be in MM only list",
                index
            ));
    }
}

/**
 * @brief find moleculeType by moleculeType if (size_t)
 *
 * @param moleculeType
 * @return Molecule
 *
 * @throw RstFileException if molecule type not found
 */
MoleculeType& SimulationBox::findMoleculeType(const size_t moleculeType)
{
    auto isMoleculeType = [moleculeType](const auto& mol)
    { return mol.getMoltype() == moleculeType; };

    const auto molecule = std::ranges::find_if(_moleculeTypes, isMoleculeType);

    if (molecule != _moleculeTypes.end())
        return *molecule;
    else
        throw RstFileException(
            std::format("Molecule type {} not found", moleculeType)
        );
}

/**
 * @brief checks if molecule type exists by moleculeType id (size_t)
 *
 * @param moleculeType
 * @return true
 * @return false
 */
bool SimulationBox::moleculeTypeExists(const size_t moleculeType) const
{
    auto isMoleculeType = [moleculeType](const auto& mol)
    { return mol.getMoltype() == moleculeType; };

    const auto molType = std::ranges::find_if(_moleculeTypes, isMoleculeType);

    return molType != _moleculeTypes.end();
}

/**
 * @brief find molecule type by string id
 *
 * @details return an optional - if moleculeType found it returns the
 * moleculeType as a size_t otherwise it returns nullopt
 *
 * @param moleculeType
 * @return optional<size_t>
 */
std::optional<size_t> SimulationBox::findMoleculeTypeByString(
    const std::string& moleculeType
) const
{
    auto isMoleculeName = [&moleculeType](const auto& mol)
    { return mol.getName() == moleculeType; };

    const auto molecule = std::ranges::find_if(_moleculeTypes, isMoleculeName);

    if (molecule != _moleculeTypes.end())
        return molecule->getMoltype();
    else
        return std::nullopt;
}

/**
 * @brief find molecule by atom index
 *
 * @details return a pair of a pointer to the molecule and the index of the atom
 * in the molecule
 *
 * @param atomIndex
 * @return pair<Molecule *, size_t>
 */
std::pair<Molecule*, size_t> SimulationBox::findMoleculeByAtomIndex(
    const size_t atomIndex
)
{
    size_t sum = 0;

    for (auto& molecule : _molecules)
    {
        const auto nAtomsInMolecule  = molecule.getNumberOfAtoms();
        sum                         += molecule.getNumberOfAtoms();

        if (sum >= atomIndex)
        {
            if (atomIndex == 0)
                break;
            const auto index = atomIndex - (sum - nAtomsInMolecule) - 1;
            return std::make_pair(&molecule, index);
        }
    }

    throw UserInputException(std::format(
        "Atom index {} out of range - total number of atoms: {}",
        atomIndex,
        sum
    ));
}

/**
 * @brief find necessary molecule types
 *
 * @details The user can specify more molecule types in the moldescriptor file
 * than actually necessary in the simulation. This function returns only the
 * molecule types which are also present in the simulation box in _molecules.
 *
 * @return std::vector<Molecule>
 */
std::vector<MoleculeType> SimulationBox::findNecessaryMoleculeTypes()
{
    std::vector<MoleculeType> neededMolTypes;

    auto searchMoleculeTypes = [&neededMolTypes, this](const auto& molecule)
    {
        auto predicate = [&molecule](const auto moleculeType)
        { return molecule.getMoltype() == moleculeType.getMoltype(); };

        const auto molType = std::ranges::find_if(neededMolTypes, predicate);

        if (molType == neededMolTypes.end() && molecule.getMoltype() != 0)
            neededMolTypes.push_back(findMoleculeType(molecule.getMoltype()));
    };

    std::ranges::for_each(_molecules, searchMoleculeTypes);

    return neededMolTypes;
}

/**
 * @brief set partial charges of molecules from molecule types
 *
 * @throw UserInputException if molecule type not found in _moleculeTypes
 *
 */
void SimulationBox::setPartialChargesOfMoleculesFromMoleculeTypes()
{
    auto setPartialCharges =
        [&moleculeTypes = _moleculeTypes](Molecule& molecule)
    {
        auto predicate = [&molecule](const auto moleculeType)
        { return molecule.getMoltype() == moleculeType.getMoltype(); };

        const auto molType = std::ranges::find_if(moleculeTypes, predicate);

        if (molType != moleculeTypes.end())
            molecule.setPartialCharges(molType->getPartialCharges());

        else if (molecule.getMoltype() != 0)
            throw UserInputException(std::format(
                "Molecule type {} not found in molecule types",
                molecule.getMoltype()
            ));
    };

    std::ranges::for_each(_molecules, setPartialCharges);
}

/**
 * @brief make external to internal global vdw types map
 *
 * @details the function consists of multiple steps:
 * 1) fill the external global vdw types vector with all external global vdw
 * types from all molecules 2) sort and erase duplicates 3) fill the external to
 * internal global vdw types map - internal vdw types are defined via increasing
 * indices 4) set the internal global vdw types for all molecules
 *
 */
void SimulationBox::setupExternalToInternalGlobalVdwTypesMap()
{
    /****************************************************************************
     * 1) fill the external global vdw types vector with all external global vdw
     * types from all molecules
     ****************************************************************************/

    auto fillExtGlobalVdwTypes =
        [&extGlobalVdwTypes = _externalGlobalVdwTypes](auto& molType)
    {
        extGlobalVdwTypes.insert(
            extGlobalVdwTypes.end(),
            molType.getExternalGlobalVDWTypes().begin(),
            molType.getExternalGlobalVDWTypes().end()
        );
    };

    std::ranges::for_each(_moleculeTypes, fillExtGlobalVdwTypes);

    /********************************
     * 2) sort and erase duplicates *
     ********************************/

    std::ranges::sort(_externalGlobalVdwTypes);
    const auto duplicates = std::ranges::unique(_externalGlobalVdwTypes);
    _externalGlobalVdwTypes.erase(duplicates.begin(), duplicates.end());

    /***********************************************************************
     * 3) fill the external to internal global vdw types map - internal vdw
     *types are defined via increasing indices
     ***********************************************************************/

    // c++23 with std::ranges::views::enumerate
    const size_t size = _externalGlobalVdwTypes.size();
    for (size_t i = 0; i < size; ++i)
    {
        const auto type = _externalGlobalVdwTypes[i];
        _externalToInternalGlobalVDWTypes.try_emplace(type, i);
    }

    /**********************************************************
     * 4) set the internal global vdw types for all molecules *
     * ********************************************************/

    auto setIntGlobalVdwTypes =
        [&extToIntGlobalVDWTypes =
             _externalToInternalGlobalVDWTypes](auto& molecule)
    {
        for (size_t i = 0; i < molecule.getNumberOfAtoms(); ++i)
        {
            const auto extType = molecule.getAtom(i).getExternalGlobalVDWType();
            molecule.getAtom(i).setInternalGlobalVDWType(
                extToIntGlobalVDWTypes.at(extType)
            );
        }
    };

    std::ranges::for_each(_molecules, setIntGlobalVdwTypes);
}

/**
 * @brief calculate degrees of freedom
 *
 */
void SimulationBox::calculateDegreesOfFreedom()
{
    const auto nAtoms = getNumberOfAtoms();

    _degreesOfFreedom = 3 * nAtoms - Settings::getDimensionality();
}

/**
 * @brief calculate total mass of simulationBox
 *
 */
void SimulationBox::calculateTotalMass()
{
    _totalMass = 0.0;

    auto accumulateMass = [this](const auto& atom)
    { _totalMass += atom->getMass(); };

    std::ranges::for_each(_atoms, accumulateMass);
}

/**
 * @brief calculate center of mass of simulationBox
 *
 */
void SimulationBox::calculateCenterOfMass()
{
    _centerOfMass = Vec3D{0.0};

    auto accumulateMassWeightedPos = [this](const auto& atom)
    { _centerOfMass += atom->getMass() * atom->getPosition(); };

    std::ranges::for_each(_atoms, accumulateMassWeightedPos);

    _centerOfMass /= _totalMass;
}

/**
 * @brief calculate center of mass of all molecules
 *
 */
void SimulationBox::calculateCenterOfMassMolecules()
{
    auto calcCenterOfMassMolecule = [this](Molecule& molecule)
    { molecule.calculateCenterOfMass(*_box); };

    std::ranges::for_each(_molecules, calcCenterOfMassMolecule);
}

/**
 * @brief calculate momentum of simulationBox
 *
 * @return Vec3D
 */
Vec3D SimulationBox::calculateMomentum()
{
    auto momentum = Vec3D{0.0};

    auto accumulateAtomicMomentum = [&momentum](const auto& atom)
    { momentum += atom->getMass() * atom->getVelocity(); };

    std::ranges::for_each(_atoms, accumulateAtomicMomentum);

    return momentum;
}

/**
 * @brief calculate angular momentum of simulationBox
 *
 */
Vec3D SimulationBox::calculateAngularMomentum(const Vec3D& momentum)
{
    auto angularMom = Vec3D{0.0};

    auto accumulateAngularMomentum = [&angularMom](const auto& atom)
    {
        const auto mass = atom->getMass();
        angularMom += mass * cross(atom->getPosition(), atom->getVelocity());
    };

    std::ranges::for_each(_atoms, accumulateAngularMomentum);

    angularMom -= cross(_centerOfMass, momentum / _totalMass) * _totalMass;

    return angularMom;
}

/**
 * @brief calculate total force of simulationBox as scalar
 *
 * @return double
 */
double SimulationBox::calculateTotalForce()
{
    const auto totalForce = calculateTotalForceVector();

    return norm(totalForce);
}

/**
 * @brief calculate total force of simulationBox as vector
 *
 * @return Vec3D
 */
Vec3D SimulationBox::calculateTotalForceVector()
{
    Vec3D totalForce(0.0);

    auto accumulateForce = [&totalForce](const auto& atom)
    { totalForce += atom->getForce(); };

    std::ranges::for_each(_atoms, accumulateForce);

    return totalForce;
}

/**
 * @brief calculate RMS force of simulationBox
 *
 * @return double
 */
double SimulationBox::calculateRMSForce() const
{
    const auto scalarForces = getAtomicScalarForces();

    return rms(scalarForces);
}

/**
 * @brief calculate max scalar force of simulationBox
 *
 * @return double
 */
double SimulationBox::calculateMaxForce() const
{
    const auto scalarForces = getAtomicScalarForces();

    return max(scalarForces);
}

/**
 * @brief calculate rms old force of simulationBox
 *
 * @return double
 */
double SimulationBox::calculateRMSForceOld() const
{
    const auto scalarForces = getAtomicScalarForcesOld();

    return rms(scalarForces);
}

/**
 * @brief calculate max force old of simulationBox
 *
 * @return double
 */
double SimulationBox::calculateMaxForceOld() const
{
    const auto scalarForces = getAtomicScalarForcesOld();

    return max(scalarForces);
}

/**
 * @brief calculate temperature of simulationBox
 *
 */
double SimulationBox::calculateTemperature()
{
    auto temperature = 0.0;

    auto accumulateTemperature = [&temperature](const auto& atom)
    { temperature += atom->getMass() * normSquared(atom->getVelocity()); };

    std::ranges::for_each(_atoms, accumulateTemperature);

    temperature *= _TEMPERATURE_FACTOR_ / double(_degreesOfFreedom);

    return temperature;
}

/**
 * @brief checks if the coulomb radius cut off is smaller than half of the
 * minimal box dimension
 *
 * @throw UserInputException if coulomb radius cut off is larger than half of
 * the minimal box dimension
 */
void SimulationBox::checkCoulRadiusCutOff(const ExceptionType exceptionType
) const
{
    const auto coulRadiusCutOff = PotentialSettings::getCoulombRadiusCutOff();

    if (getMinimalBoxDimension() < 2.0 * coulRadiusCutOff)
    {
        const std::string message =
            "Coulomb radius cut off is larger than half of the minimal box "
            "dimension";

        if (exceptionType == ExceptionType::MANOSTATEXCEPTION)
            throw ManostatException(message);
        else
            throw UserInputException(message);
    }
}

/**
 * @brief return all unique qm atom names
 *
 * @return std::vector<std::string>
 */
std::vector<std::string> SimulationBox::getUniqueQMAtomNames()
{
    std::vector<std::string> uniqueQMAtomNames;

    auto fillQMAtomNames = std::back_inserter(uniqueQMAtomNames);
    auto getName         = [](const auto& atom) { return atom->getName(); };

    std::ranges::transform(_qmAtoms, fillQMAtomNames, getName);
    std::ranges::sort(uniqueQMAtomNames);

    const auto [first, last] = std::ranges::unique(uniqueQMAtomNames);

    uniqueQMAtomNames.erase(first, last);

    return uniqueQMAtomNames;
}

/**
 * @brief calculate density of simulationBox
 *
 */
void SimulationBox::calculateDensity()
{
    const auto volume = _box->calculateVolume();
    _density          = _totalMass / volume * _AMU_PER_ANGSTROM3_TO_KG_PER_L_;
}

/**
 * @brief calculate box dimensions from density
 *
 * @return Vec3D
 */
Vec3D SimulationBox::calcBoxDimFromDensity() const
{
    auto& orthoBox = dynamic_cast<OrthorhombicBox&>(*_box);
    return orthoBox.calcBoxDimFromDensity(_totalMass, _density);
}

/**
 * @brief calculate shift vector
 *
 * @param position
 * @return Vec3D
 */
Vec3D SimulationBox::calcShiftVector(const Vec3D& position) const
{
    return _box->calcShiftVector(position);
}

/**
 * @brief initialize positions of all atoms
 *
 */
void SimulationBox::initPositions(const double displacement)
{
    RandomNumberGenerator randomNumberGenerator{};

    auto displacePositions =
        [&randomNumberGenerator, displacement, this](auto& atom)
    {
        const auto random = Vec3D{
            randomNumberGenerator
                .getUniformRealDistribution(-displacement, displacement),
            randomNumberGenerator
                .getUniformRealDistribution(-displacement, displacement),
            randomNumberGenerator
                .getUniformRealDistribution(-displacement, displacement)
        };

        auto position = atom->getPosition() + random;

        applyPBC(position);

        atom->setPosition(position);
    };

    std::ranges::for_each(_atoms, displacePositions);
}

/**
 * @brief update old positions of all atoms
 *
 */
void SimulationBox::updateOldPositions()
{
    auto updateOldPosition = [](const auto& atom)
    { atom->updateOldPosition(); };

    std::ranges::for_each(_atoms, updateOldPosition);
}

/**
 * @brief update old velocities of all atoms
 *
 */
void SimulationBox::updateOldVelocities()
{
    auto updateOldVelocity = [](const auto& atom)
    { atom->updateOldVelocity(); };

    std::ranges::for_each(_atoms, updateOldVelocity);
}

/**
 * @brief update old forces of all atoms
 *
 */
void SimulationBox::updateOldForces()
{
    auto updateOldForce = [](const auto& atom) { atom->updateOldForce(); };

    std::ranges::for_each(_atoms, updateOldForce);
}

/**
 * @brief reset forces of all atoms
 *
 */
void SimulationBox::resetForces()
{
    auto resetForces = [](const auto& atom) { atom->setForceToZero(); };

    std::ranges::for_each(_atoms, resetForces);
}