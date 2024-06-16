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
#include "settings.hpp"            // for Settings
#include "stlVector.hpp"           // for rms

using simulationBox::Molecule;
using simulationBox::MoleculeType;
using simulationBox::SimulationBox;

/**
 * @brief copy simulationBox object this
 *
 * @details shared_ptrs are not copied but new ones are created
 *
 * @notes copy constructor is not used because it would break semantics here
 *
 * @param toCopy
 */
void SimulationBox::copy(const SimulationBox &toCopy)
{
    *this = toCopy;

    this->_atoms.clear();
    this->_qmAtoms.clear();
    for (size_t i = 0; i < toCopy._atoms.size(); ++i)
    {
        const auto atom = std::make_shared<Atom>(*toCopy._atoms[i]);
        this->_atoms.push_back(atom);
        // TODO: ATTENTION AT THE MOMENT ONLY VALID FOR ALL QM_CALCULATIONS
        this->_qmAtoms.push_back(atom);
    }

    auto fillAtomsInMolecules = [this](size_t runningIndex, Molecule &molecule)
    {
        molecule.getAtoms().clear();
        for (size_t i = 0; i < molecule.getNumberOfAtoms(); ++i)
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
 * @brief finds molecule by moleculeType if (size_t)
 *
 * @param moleculeType
 * @return std::optional<Molecule &>
 */
std::optional<Molecule> SimulationBox::findMolecule(const size_t moleculeType)
{
    auto isMoleculeType = [moleculeType](const Molecule &mol)
    { return mol.getMoltype() == moleculeType; };

    if (const auto molecule = std::ranges::find_if(_molecules, isMoleculeType);
        molecule != _molecules.end())
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
void SimulationBox::addQMCenterAtoms(const std::vector<int> &atomIndices)
{
    for (const auto index : atomIndices)
    {
        if (index < 0 || (size_t) index >= _atoms.size())
            throw customException::UserInputException(
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
void SimulationBox::setupQMOnlyAtoms(const std::vector<int> &atomIndices)
{
    for (const auto index : atomIndices)
    {
        if (index < 0 || (size_t) index >= _atoms.size())
            throw customException::UserInputException(
                std::format("QM only atom index {} out of range", index)
            );

        _atoms[(size_t) index]->setQMOnly(true);

        if (std::ranges::find(
                _qmAtoms.begin(),
                _qmAtoms.end(),
                _atoms[(size_t) index]
            ) == _qmAtoms.end())
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
void SimulationBox::setupMMOnlyAtoms(const std::vector<int> &atomIndices)
{
    for (const auto index : atomIndices)
    {
        if (index < 0 || (size_t) index >= _atoms.size())
            throw customException::UserInputException(
                std::format("MM only atom index {} out of range", index)
            );

        _atoms[(size_t) index]->setMMOnly(true);

        if (std::ranges::find(
                _qmAtoms.begin(),
                _qmAtoms.end(),
                _atoms[(size_t) index]
            ) != _qmAtoms.end())

            throw customException::UserInputException(std::format(
                "Ambiguous atom index {} - atom is already in QM only list - "
                "cannot be in MM only "
                "list",
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
MoleculeType &SimulationBox::findMoleculeType(const size_t moleculeType)
{
    auto isMoleculeType = [moleculeType](const auto &mol)
    { return mol.getMoltype() == moleculeType; };

    if (const auto molecule =
            std::ranges::find_if(_moleculeTypes, isMoleculeType);
        molecule != _moleculeTypes.end())
        return *molecule;
    else
        throw customException::RstFileException(
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
    auto isMoleculeType = [moleculeType](const auto &mol)
    { return mol.getMoltype() == moleculeType; };

    return std::ranges::find_if(_moleculeTypes, isMoleculeType) !=
           _moleculeTypes.end();
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
    const std::string &moleculeType
) const
{
    auto isMoleculeName = [&moleculeType](const auto &mol)
    { return mol.getName() == moleculeType; };

    if (const auto molecule =
            std::ranges::find_if(_moleculeTypes, isMoleculeName);
        molecule != _moleculeTypes.end())
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
std::pair<Molecule *, size_t> SimulationBox::findMoleculeByAtomIndex(
    const size_t atomIndex
)
{
    size_t sum = 0;

    for (auto &molecule : _molecules)
    {
        sum += molecule.getNumberOfAtoms();

        if (sum >= atomIndex)
            return std::make_pair(
                &molecule,
                atomIndex - (sum - molecule.getNumberOfAtoms()) - 1
            );
    }

    throw customException::UserInputException(std::format(
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
    std::vector<MoleculeType> necessaryMoleculeTypes;

    auto searchMoleculeTypes =
        [&necessaryMoleculeTypes, this](const auto &molecule)
    {
        auto predicate = [&molecule](const auto moleculeType)
        { return molecule.getMoltype() == moleculeType.getMoltype(); };

        if (std::ranges::find_if(necessaryMoleculeTypes, predicate) ==
            necessaryMoleculeTypes.end())
            if (molecule.getMoltype() != 0)
                necessaryMoleculeTypes.push_back(
                    findMoleculeType(molecule.getMoltype())
                );
    };

    std::ranges::for_each(_molecules, searchMoleculeTypes);

    return necessaryMoleculeTypes;
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
        [&moleculeTypes = _moleculeTypes](Molecule &molecule)
    {
        auto predicate = [&molecule](const auto moleculeType)
        { return molecule.getMoltype() == moleculeType.getMoltype(); };

        if (const auto moleculeType =
                std::ranges::find_if(moleculeTypes, predicate);
            moleculeType != moleculeTypes.end())
            molecule.setPartialCharges(moleculeType->getPartialCharges());
        else if (molecule.getMoltype() != 0)
            throw customException::UserInputException(std::format(
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
    /******************************************************************************************************
     * 1) fill the external global vdw types vector with all external global vdw
     *types from all molecules *
     ******************************************************************************************************/

    auto fillExternalGlobalVdwTypes =
        [&externalGlobalVdwTypes = _externalGlobalVdwTypes](auto &moleculeType)
    {
        externalGlobalVdwTypes.insert(
            externalGlobalVdwTypes.end(),
            moleculeType.getExternalGlobalVDWTypes().begin(),
            moleculeType.getExternalGlobalVDWTypes().end()
        );
    };

    std::ranges::for_each(_moleculeTypes, fillExternalGlobalVdwTypes);

    /********************************
     * 2) sort and erase duplicates *
     *******************************/

    std::ranges::sort(_externalGlobalVdwTypes);
    const auto duplicates = std::ranges::unique(_externalGlobalVdwTypes);
    _externalGlobalVdwTypes.erase(duplicates.begin(), duplicates.end());

    /*****************************************************************************************************************
     * 3) fill the external to internal global vdw types map - internal vdw
     *types are defined via increasing indices *
     *****************************************************************************************************************/

    // c++23 with std::ranges::views::enumerate
    const size_t size = _externalGlobalVdwTypes.size();
    for (size_t i = 0; i < size; ++i)
        _externalToInternalGlobalVDWTypes.try_emplace(
            _externalGlobalVdwTypes[i],
            i
        );

    /**********************************************************
     * 4) set the internal global vdw types for all molecules *
     * ********************************************************/

    auto setInternalGlobalVdwTypes =
        [&externalToInternalGlobalVDWTypes =
             _externalToInternalGlobalVDWTypes](auto &molecule)
    {
        for (size_t i = 0; i < molecule.getNumberOfAtoms(); ++i)
            molecule.getAtom(i).setInternalGlobalVDWType(
                externalToInternalGlobalVDWTypes.at(
                    molecule.getAtom(i).getExternalGlobalVDWType()
                )
            );
    };

    std::ranges::for_each(_molecules, setInternalGlobalVdwTypes);
}

/**
 * @brief calculate degrees of freedom
 *
 */
void SimulationBox::calculateDegreesOfFreedom()
{
    auto accumulateFunc = [](const size_t sum, const Molecule &molecule)
    { return sum + molecule.getDegreesOfFreedom(); };

    _degreesOfFreedom =
        accumulate(_molecules.begin(), _molecules.end(), 0UL, accumulateFunc) -
        settings::Settings::getDimensionality();
}

/**
 * @brief calculate total mass of simulationBox
 *
 */
void SimulationBox::calculateTotalMass()
{
    _totalMass = 0.0;

    std::ranges::for_each(
        _atoms,
        [this](const auto &atom) { _totalMass += atom->getMass(); }
    );
}

/**
 * @brief calculate center of mass of simulationBox
 *
 */
void SimulationBox::calculateCenterOfMass()
{
    _centerOfMass = linearAlgebra::Vec3D{0.0};

    std::ranges::for_each(
        _atoms,
        [this](const auto &atom)
        { _centerOfMass += atom->getMass() * atom->getPosition(); }
    );

    _centerOfMass /= _totalMass;
}

/**
 * @brief calculate center of mass of all molecules
 *
 */
void SimulationBox::calculateCenterOfMassMolecules()
{
    std::ranges::for_each(
        _molecules,
        [&box = _box](Molecule &molecule)
        { molecule.calculateCenterOfMass(*box); }
    );
}

/**
 * @brief calculate momentum of simulationBox
 *
 */
linearAlgebra::Vec3D SimulationBox::calculateMomentum()
{
    auto momentum = linearAlgebra::Vec3D{0.0};

    std::ranges::for_each(
        _atoms,
        [&momentum](const auto &atom)
        { momentum += atom->getMass() * atom->getVelocity(); }
    );

    return momentum;
}

/**
 * @brief calculate angular momentum of simulationBox
 *
 */
linearAlgebra::Vec3D SimulationBox::calculateAngularMomentum(
    const linearAlgebra::Vec3D &momentum
)
{
    auto angularMomentum = linearAlgebra::Vec3D{0.0};

    std::ranges::for_each(
        _atoms,
        [&angularMomentum](const auto &atom)
        {
            angularMomentum += atom->getMass() *
                               cross(atom->getPosition(), atom->getVelocity());
        }
    );

    angularMomentum -= cross(_centerOfMass, momentum / _totalMass) * _totalMass;

    return angularMomentum;
}

/**
 * @brief calculate total force of simulationBox
 *
 * @return double
 */
double SimulationBox::calculateTotalForce()
{
    linearAlgebra::Vec3D totalForce(0.0);

    std::ranges::for_each(
        _atoms,
        [&totalForce](const auto &atom) { totalForce += atom->getForce(); }
    );

    return norm(totalForce);
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
 * @brief calculate temperature of simulationBox
 *
 */
double SimulationBox::calculateTemperature()
{
    auto temperature = 0.0;

    std::ranges::for_each(
        _atoms,
        [&temperature](const auto &atom)
        { temperature += atom->getMass() * normSquared(atom->getVelocity()); }
    );

    temperature *= constants::_TEMPERATURE_FACTOR_ / double(_degreesOfFreedom);

    return temperature;
}

/**
 * @brief checks if the coulomb radius cut off is smaller than half of the
 * minimal box dimension
 *
 * @throw UserInputException if coulomb radius cut off is larger than half of
 * the minimal box dimension
 */
void SimulationBox::checkCoulombRadiusCutOff(
    const customException::ExceptionType exceptionType
) const
{
    if (getMinimalBoxDimension() <
        2.0 * settings::PotentialSettings::getCoulombRadiusCutOff())
    {
        const std::string message =
            "Coulomb radius cut off is larger than half of the minimal box "
            "dimension";
        if (exceptionType == customException::ExceptionType::MANOSTATEXCEPTION)
            throw customException::ManostatException(message);
        else
            throw customException::UserInputException(message);
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

    std::ranges::transform(
        _qmAtoms,
        std::back_inserter(uniqueQMAtomNames),
        [](const auto &atom) { return atom->getName(); }
    );
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
    _density = _totalMass / _box->calculateVolume() *
               constants::_AMU_PER_ANGSTROM3_TO_KG_PER_L_;
}

/**
 * @brief calculate box dimensions from density
 *
 * @return linearAlgebra::Vec3D
 */
linearAlgebra::Vec3D SimulationBox::calculateBoxDimensionsFromDensity() const
{
    return dynamic_cast<OrthorhombicBox &>(*_box)
        .calculateBoxDimensionsFromDensity(_totalMass, _density);
}

/**
 * @brief calculate shift vector
 *
 * @param position
 * @return linearAlgebra::Vec3D
 */
linearAlgebra::Vec3D SimulationBox::calculateShiftVector(
    const linearAlgebra::Vec3D &position
) const
{
    return _box->calculateShiftVector(position);
}

/**
 * @brief initialize positions of all atoms
 *
 */
void SimulationBox::initPositions(const double displacement)
{
    std::random_device               randomDevice;
    std::mt19937                     randomGenerator(randomDevice());
    std::uniform_real_distribution<> uniformDistribution(
        -displacement,
        displacement
    );

    auto displacePositions =
        [&uniformDistribution, &randomGenerator, this](auto &atom)
    {
        auto position =
            atom->getPosition() + linearAlgebra::Vec3D{
                                      uniformDistribution(randomGenerator),
                                      uniformDistribution(randomGenerator),
                                      uniformDistribution(randomGenerator)
                                  };

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
    std::ranges::for_each(
        _atoms,
        [](const auto &atom) { atom->updateOldPosition(); }
    );
}

/**
 * @brief update old velocities of all atoms
 *
 */
void SimulationBox::updateOldVelocities()
{
    std::ranges::for_each(
        _atoms,
        [](const auto &atom) { atom->updateOldVelocity(); }
    );
}

/**
 * @brief update old forces of all atoms
 *
 */
void SimulationBox::updateOldForces()
{
    std::ranges::for_each(
        _atoms,
        [](const auto &atom) { atom->updateOldForce(); }
    );
}