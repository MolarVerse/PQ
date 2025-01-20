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
#include <vector>

#include "constants.hpp"           // for _TEMPERATURE_FACTOR_
#include "exceptions.hpp"          // for RstFileException, UserInputException
#include "linearAlgebra.hpp"       // for Vec3D
#include "potentialSettings.hpp"   // for PotentialSettings
#include "settings.hpp"            // for Settings

using namespace linearAlgebra;
using namespace simulationBox;
using namespace customException;
using namespace constants;
using namespace settings;

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
        {
            molecule.addAtom(this->_atoms[runningIndex]);
            ++runningIndex;
        }

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
 * @brief resize host vectors
 * @param nAtoms
 * @param nMolecules
 */
void SimulationBox::resizeHostVectors(
    const size_t nAtoms,
    const size_t nMolecules
)
{
    _Coordinates::resizeHostVectors(nAtoms, nMolecules);
    _SimulationBoxSoA::resizeHostVectors(nAtoms, nMolecules);
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
 * @brief calculate RMS force of simulationBox
 *
 * @return double
 */
double SimulationBox::calculateRMSForce()
{
    const auto scalarForces = getAtomicScalarForces();

    return rms(scalarForces);
}

/**
 * @brief calculate max scalar force of simulationBox
 *
 * @return double
 */
double SimulationBox::calculateMaxForce()
{
    const auto scalarForces = getAtomicScalarForces();

    return max(scalarForces);
}

/**
 * @brief calculate rms old force of simulationBox
 *
 * @return double
 */
double SimulationBox::calculateRMSForceOld()
{
    const auto scalarForces = getAtomicScalarForcesOld();

    return rms(scalarForces);
}

/**
 * @brief calculate max force old of simulationBox
 *
 * @return double
 */
double SimulationBox::calculateMaxForceOld()
{
    const auto scalarForces = getAtomicScalarForcesOld();

    return max(scalarForces);
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
    std::random_device             randomDevice;
    std::mt19937                   randomGenerator(randomDevice());
    std::uniform_real_distribution uniformDist{-displacement, displacement};

    auto displacePositions = [&uniformDist, &randomGenerator, this](auto& atom)
    {
        const auto random = Vec3D{
            uniformDist(randomGenerator),
            uniformDist(randomGenerator),
            uniformDist(randomGenerator)
        };

        auto position = atom->getPosition() + random;

        applyPBC(position);

        atom->setPosition(position);
    };

    std::ranges::for_each(_atoms, displacePositions);
}

/**
 * @brief flattens positions of each atom into a single vector of doubles
 *
 */
void SimulationBox::flattenPositions()
{
    if (_pos.size() != _atoms.size() * 3)
    {
        _pos.resize(_atoms.size() * 3);
    }

    // clang-format off
    #pragma omp parallel for
    // clang-format on
    for (size_t i = 0; i < _atoms.size(); ++i)
    {
        const auto atom = _atoms[i];

        for (size_t j = 0; j < 3; ++j)
            _pos[i * 3 + j] = atom->getPosition()[j];
    }

#ifdef __PQ_GPU__
    if (Settings::useDevice())
        copyPosTo();
#endif
}

/**
 * @brief de-flattens positions of each atom from a single vector of doubles
 * into the atom objects
 *
 * @param positions
 */
void SimulationBox::deFlattenPositions()
{
#ifdef __PQ_GPU__
    if (Settings::useDevice())
    {
        copyPosFrom();
    }
#endif

    // clang-format off
    #pragma omp parallel for
    // clang-format on
    for (size_t i = 0; i < _atoms.size(); ++i)
    {
        const auto atom = _atoms[i];

        for (size_t j = 0; j < 3; ++j)
            atom->setPosition(_pos[i * 3 + j], j);
    }
}

/**
 * @brief flattens velocities of each atom into a single vector of doubles
 *
 */
void SimulationBox::flattenVelocities()
{
    if (_vel.size() != _atoms.size() * 3)
    {
        _vel.resize(_atoms.size() * 3);
    }

    // clang-format off
    #pragma omp parallel for
    // clang-format on
    for (size_t i = 0; i < _atoms.size(); ++i)
    {
        const auto atom = _atoms[i];

        for (size_t j = 0; j < 3; ++j)
            _vel[i * 3 + j] = atom->getVelocity()[j];
    }

#ifdef __PQ_GPU__
    if (Settings::useDevice())
        copyVelTo();
#endif
}

/**
 * @brief de-flattens velocities of each atom from a single vector of doubles
 * into the atom objects
 *
 * @param velocities
 */
void SimulationBox::deFlattenVelocities()
{
#ifdef __PQ_GPU__
    if (Settings::useDevice())
    {
        copyVelFrom();
    }
#endif

    // clang-format off
    #pragma omp parallel for
    // clang-format on
    for (size_t i = 0; i < _atoms.size(); ++i)
    {
        const auto atom = _atoms[i];

        for (size_t j = 0; j < 3; ++j)
            atom->setVelocity(_vel[i * 3 + j], j);
    }
}

/**
 * @brief flattens forces of each atom into a single vector of doubles
 *
 */
void SimulationBox::flattenForces()
{
    if (_forces.size() != _atoms.size() * 3)
    {
        _forces.resize(_atoms.size() * 3);
    }

    // clang-format off
    #pragma omp parallel for
    // clang-format on
    for (size_t i = 0; i < _atoms.size(); ++i)
    {
        const auto atom = _atoms[i];

        for (size_t j = 0; j < 3; ++j)
            _forces[i * 3 + j] = atom->getForce()[j];
    }

#ifdef __PQ_GPU__
    if (Settings::useDevice())
        copyForcesTo();
#endif
}

/**
 * @brief de-flattens forces of each atom from a single vector of doubles into
 * the atom objects
 *
 * @param forces
 */
void SimulationBox::deFlattenForces()
{
#ifdef __PQ_GPU__
    if (Settings::useDevice())
    {
        copyForcesFrom();
    }
#endif

    // clang-format off
    #pragma omp parallel for
    // clang-format on
    for (size_t i = 0; i < _atoms.size(); ++i)
    {
        const auto atom = _atoms[i];

        for (size_t j = 0; j < 3; ++j)
            atom->setForce(_forces[i * 3 + j], j);
    }
}

/**
 * @brief flattens shift forces of each atom into a single vector of doubles
 *
 * @return std::vector<double>
 */
void SimulationBox::flattenShiftForces()
{
    if (_shiftForces.size() != _atoms.size() * 3)
        _shiftForces.resize(_atoms.size() * 3);

    Real* const shiftForces = _shiftForces.data();

    // clang-format off
    #pragma omp parallel for
    // clang-format on
    for (size_t i = 0; i < _atoms.size(); ++i)
    {
        const auto atom = _atoms[i];

        for (size_t j = 0; j < 3; ++j)
            shiftForces[i * 3 + j] = atom->getShiftForce()[j];
    }

#ifdef __PQ_GPU__
    if (Settings::useDevice())
        copyShiftForcesTo();
#endif
}

/**
 * @brief de-flattens shift forces of each atom from a single vector of doubles
 * into the atom objects
 *
 * @param shiftForces
 */
void SimulationBox::deFlattenShiftForces()
{
#ifdef __PQ_GPU__
    if (Settings::useDevice())
        copyShiftForcesFrom();
#endif

    Real* const shiftForces = _shiftForces.data();

    // clang-format off
    #pragma omp parallel for
    // clang-format on
    for (size_t i = 0; i < _atoms.size(); ++i)
    {
        const auto atom = _atoms[i];

        for (size_t j = 0; j < 3; ++j)
            atom->setShiftForce(shiftForces[i * 3 + j], j);
    }
}

/**
 * @brief flattens masses of each atom into a single vector of Real
 */
void SimulationBox::flattenMasses()
{
    if (_masses.size() != _atoms.size())
        _masses.resize(_atoms.size());

        // clang-format off
    #pragma omp parallel for
    // clang-format on
    for (size_t i = 0; i < _atoms.size(); ++i)
        _masses[i] = _atoms[i]->getMass();

#ifdef __PQ_GPU__
    if (Settings::useDevice())
        copyMassesTo();
#endif
}

/**
 * @brief flattens mol masses of each molecule into a single vector of Real
 *
 */
void SimulationBox::flattenMolMasses()
{
    if (_molMasses.size() != _molecules.size())
        _molMasses.resize(_molecules.size());

    const auto nMolecules = _molecules.size();

    // clang-format off
    #pragma omp parallel for
    // clang-format on
    for (size_t i = 0; i < nMolecules; ++i)
        _molMasses[i] = _molecules[i].getMolMass();

#ifdef __PQ_GPU__
    if (Settings::useDevice())
        copyMolMassesTo();
#endif
}

/**
 * @brief flattens charges of each atom into a single vector of Real
 *
 */
void SimulationBox::flattenCharges()
{
    if (_charges.size() != _atoms.size())
        _charges.resize(_atoms.size());

        // clang-format off
    #pragma omp parallel for
    // clang-format on
    for (size_t i = 0; i < _atoms.size(); ++i)
        _charges[i] = _atoms[i]->getPartialCharge();

#ifdef __PQ_GPU__
    if (Settings::useDevice())
        copyChargesTo();
#endif
}

/**
 * @brief flattens center of mass of molecules of each molecule into a single
 * vector of Vec3D
 *
 */
void SimulationBox::flattenComMolecules()
{
    if (_comMolecules.size() != _molecules.size() * 3)
        _comMolecules.resize(_molecules.size() * 3);

        // clang-format off
    #pragma omp parallel for
    // clang-format on
    for (size_t i = 0; i < _molecules.size(); ++i)
    {
        const auto com                   = _molecules[i].getCenterOfMass();
        const auto moleculeIndex         = i * 3;
        _comMolecules[moleculeIndex]     = com[0];
        _comMolecules[moleculeIndex + 1] = com[1];
        _comMolecules[moleculeIndex + 2] = com[2];
    }

#ifdef __PQ_GPU__
    if (Settings::useDevice())
        copyComMoleculesTo();
#endif
}

/**
 * @brief de-flattens center of mass of molecules of each molecule from a single
 * vector of Vec3D into the molecule objects
 *
 * @param comMolecules
 */
void SimulationBox::deFlattenComMolecules()
{
#ifdef __PQ_GPU__
    if (Settings::useDevice())
    {
        copyComMoleculesFrom();
    }
#endif

    // clang-format off
    #pragma omp parallel for
    // clang-format on
    for (size_t i = 0; i < _molecules.size(); ++i)
    {
        auto&      molecule      = _molecules[i];
        const auto moleculeIndex = i * 3;

        molecule.setCenterOfMass(Vec3D(
            _comMolecules[moleculeIndex],
            _comMolecules[moleculeIndex + 1],
            _comMolecules[moleculeIndex + 2]
        ));
    }
}

/**
 * @brief initializes a vector that is n molecules long and each element
 * contains the number of atoms in the molecule
 *
 */
void SimulationBox::initAtomsPerMolecule()
{
    if (_atomsPerMolecule.size() != _molecules.size())
        _atomsPerMolecule.resize(_molecules.size());

        // clang-format off
    #pragma omp parallel for
    // clang-format on
    for (size_t i = 0; i < _molecules.size(); ++i)
        _atomsPerMolecule[i] = _molecules[i].getNumberOfAtoms();

#ifdef __PQ_GPU__
    if (Settings::useDevice())
        copyAtomsPerMoleculeTo();
#endif
}

/**
 * @brief initializes a vector that is n atoms long and each element contains
 * the index of the molecule to which the atom belongs
 */
void SimulationBox::initMoleculeIndices()
{
    if (_moleculeIndices.size() != _atoms.size())
        _moleculeIndices.resize(_atoms.size());

    size_t     atomIndex  = 0;
    const auto nMolecules = _molecules.size();

    for (size_t i = 0; i < nMolecules; ++i)
    {
        const auto nAtoms = _molecules[i].getNumberOfAtoms();

        for (size_t j = 0; j < nAtoms; ++j)
        {
            _moleculeIndices[atomIndex] = i;
            ++atomIndex;
        }
    }

#ifdef __PQ_GPU__
    if (Settings::useDevice())
        copyMoleculeIndicesTo();
#endif
}

/**
 * @brief initializes a vector that is n molecules long and each element
 * contains the offset of the molecule in the atom vector
 *
 */

void SimulationBox::initMoleculeOffsets()
{
    if (_moleculeOffsets.size() != _molecules.size())
        _moleculeOffsets.resize(_molecules.size());

    size_t offset = 0;

    for (size_t i = 0; i < _molecules.size(); ++i)
    {
        _moleculeOffsets[i]  = offset;
        offset              += _molecules[i].getNumberOfAtoms();
    }

#ifdef __PQ_GPU__
    if (Settings::useDevice())
        copyMoleculeOffsetsTo();
#endif
}

/**
 * @brief scale all velocities by a factor
 *
 * @param lambda
 */
void SimulationBox::scaleVelocities(const Real lambda)
{
    Real* const _velPtr = getVelPtr();

    // clang-format off
#ifdef __PQ_GPU__
    #pragma omp target teams distribute parallel for \
                is_device_ptr(_velPtr)
#else
    #pragma omp parallel for
#endif
    // clang-format on
    for (size_t i = 0; i < _nAtoms * 3; ++i)
        _velPtr[i] *= lambda;

#ifdef __PQ_LEGACY__
    deFlattenVelocities();
#endif
}

/**
 * @brief add to all velocities a vector
 *
 * @param velocity
 */
void SimulationBox::addToVelocities(const Vec3D& toAdd)
{
    Real* const velPtr      = getVelPtr();
    const Real  toAddPtr[3] = {toAdd[0], toAdd[1], toAdd[2]};

    // clang-format off
#ifdef __PQ_GPU__
    #pragma omp target teams distribute parallel for \
                collapse(2)                          \
                is_device_ptr(velPtr)                \
                map(toAddPtr)
#else
    #pragma omp parallel for                         \
                collapse(2)
#endif
    // clang-format on
    for (size_t i = 0; i < _nAtoms; ++i)
        for (size_t j = 0; j < 3; ++j)
            velPtr[i * 3 + j] += toAddPtr[j];

#ifdef __PQ_LEGACY__
    deFlattenVelocities();
#endif
}