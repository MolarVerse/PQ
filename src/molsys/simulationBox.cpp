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

#include <algorithm>
#include <format>
#include <numeric>

#include "exceptions.hpp"
#include "generalSettings.hpp"
#include "orthorhombicBox.hpp"
#include "randomNumberGenerator.hpp"
#include "stlVector.hpp"
#include "unit_types.hpp"
#include "units.hpp"
#include "vector3d/vector3dClass.hpp"

namespace molsys
{

    /**
     * @brief copy simulationBox object this
     *
     * @details shared_ptrs are not copied but new ones are created,
     * additionally, copy constructor is not used because it would break
     * semantics here
     *
     * @param other
     */
    void SimulationBox::copy(const SimulationBox& other)
    {
        *this = other;

        this->_atoms.clear();

        for (const auto& _atom : other._atoms)
        {
            const auto atom = std::make_shared<Atom>(*_atom);
            this->_atoms.push_back(atom);
        }

        auto fillAtomsInMolecules =
            [this](size_t runningIndex, Molecule& molecule)
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
     * @brief finds molecule by moleculeType if size_t
     *
     * @param moleculeType
     * @return std::optional<Molecule &>
     */
    std::optional<Molecule> SimulationBox::findMolecule(MolType moleculeType)
    {
        auto isMoleculeType = [moleculeType](const Molecule& mol)
        { return mol.getMoltype() == moleculeType; };

        const auto molecule = std::ranges::find_if(_molecules, isMoleculeType);

        if (molecule != _molecules.end())
            return *molecule;

        return std::nullopt;
    }

    /**
     * @brief marks molecules with the given indices as forced-core molecules
     *
     * The forced-core, forced-layer, and forced-outer regions are mutually
     * exclusive. A molecule that is already assigned to either of the other
     * forced regions is rejected.
     *
     * @param moleculeIndices indices of molecules to assign to the forced-core
     * region
     *
     * @throw exc::UserInputException if molecule index is out of range
     * @throw exc::UserInputException if the molecule is already assigned to the
     * forced-layer or forced-outer region
     */
    void SimulationBox::setupForcedCoreMolecules(
        const std::vector<int>& moleculeIndices
    )
    {
        for (const auto index : moleculeIndices)
        {
            if (index < 0 || index >= static_cast<int>(_molecules.size()))
            {
                throw exc::UserInputException(
                    std::format(
                        "Forced CORE region molecule index {} out of range",
                        index
                    )
                );
            }

            auto& molecule = _molecules[static_cast<size_t>(index)];

            if (molecule.isForcedOuter() || molecule.isForcedLayer())
            {
                throw exc::UserInputException(
                    std::format(
                        "Ambiguous molecule index {} - molecule cannot be in "
                        "forced_core_list AND "
                        "forced_layer_list/forced_outer_list "
                        "at the same time",
                        index
                    )
                );
            }
            molecule.setForcedCore(true);
        }
    }

    /**
     * @brief marks molecules with the given indices as forced-layer molecules
     *
     * The forced-core, forced-layer, and forced-outer regions are mutually
     * exclusive. A molecule that is already assigned to either of the other
     * forced regions is rejected.
     *
     * @param moleculeIndices indices of molecules to assign to the forced-layer
     * region
     *
     * @throw exc::UserInputException if molecule index is out of range
     * @throw exc::UserInputException if the molecule is already assigned to the
     * forced-core or forced-outer region
     */
    void SimulationBox::setupForcedLayerMolecules(
        const std::vector<int>& moleculeIndices
    )
    {
        for (const auto index : moleculeIndices)
        {
            if (index < 0 || index >= static_cast<int>(_molecules.size()))
            {
                throw exc::UserInputException(
                    std::format(
                        "Forced Layer region molecule index {} out of range",
                        index
                    )
                );
            }

            auto& molecule = _molecules[static_cast<size_t>(index)];

            if (molecule.isForcedCore() || molecule.isForcedOuter())
            {
                throw exc::UserInputException(
                    std::format(
                        "Ambiguous molecule index {} - molecule cannot be in "
                        "forced_layer_list AND "
                        "forced_core_list/forced_outer_list "
                        "at the same time",
                        index
                    )
                );
            }

            molecule.setForcedLayer(true);
        }
    }

    /**
     * @brief marks molecules with the given indices as forced-outer molecules
     *
     * The forced-core, forced-layer, and forced-outer regions are mutually
     * exclusive. A molecule that is already assigned to either of the other
     * forced regions is rejected.
     *
     * @param moleculeIndices indices of molecules to assign to the forced-outer
     * region
     *
     * @throw exc::UserInputException if molecule index is out of range
     * @throw exc::UserInputException if the molecule is already assigned to the
     * forced-core or forced-layer region
     */
    void SimulationBox::setupForcedOuterMolecules(
        const std::vector<int>& moleculeIndices
    )
    {
        for (const auto index : moleculeIndices)
        {
            if (index < 0 || index >= static_cast<int>(_molecules.size()))
            {
                throw exc::UserInputException(
                    std::format(
                        "Forced outer region molecule index {} out of range",
                        index
                    )
                );
            }

            auto& molecule = _molecules[static_cast<size_t>(index)];

            if (molecule.isForcedCore() || molecule.isForcedLayer())
            {
                throw exc::UserInputException(
                    std::format(
                        "Ambiguous molecule index {} - molecule cannot be in "
                        "forced_outer_list AND "
                        "forced_core_list/forced_layer_list "
                        "at the same time",
                        index
                    )
                );
            }
            molecule.setForcedOuter(true);
        }
    }

    /**
     * @brief find moleculeType by moleculeType if size_t
     *
     * @param molType
     * @return Molecule
     *
     * @throw exc::RstFileException if molecule type not found
     */
    MoleculeType& SimulationBox::findMoleculeType(MolType molType)
    {
        auto isMoleculeType = [molType](const auto& mol)
        { return mol.getMoltype() == molType; };

        const auto molecule =
            std::ranges::find_if(_moleculeTypes, isMoleculeType);

        if (molecule != _moleculeTypes.end())
            return *molecule;

        throw exc::RstFileException(
            std::format("Molecule type {} not found", molType.toString())
        );
    }

    /**
     * @brief checks if molecule type exists by moleculeType id size_t
     *
     * @param moleculeType
     * @return true
     * @return false
     */
    bool SimulationBox::moleculeTypeExists(MolType moleculeType) const
    {
        auto isMoleculeType = [moleculeType](const auto& mol)
        { return mol.getMoltype() == moleculeType; };

        const auto molType =
            std::ranges::find_if(_moleculeTypes, isMoleculeType);

        return molType != _moleculeTypes.end();
    }

    /**
     * @brief find molecule type by string id
     *
     * @details return an optional - if moleculeType found it returns the
     * moleculeType as a MolType otherwise it returns nullopt
     *
     * @param moleculeType
     * @return std::optional<MolType>
     */
    std::optional<MolType> SimulationBox::findMoleculeTypeByString(
        const std::string& moleculeType
    ) const
    {
        auto isMoleculeName = [&moleculeType](const auto& mol)
        { return mol.getName() == moleculeType; };

        const auto molecule =
            std::ranges::find_if(_moleculeTypes, isMoleculeName);

        if (molecule != _moleculeTypes.end())
            return molecule->getMoltype();
        return std::nullopt;
    }

    /**
     * @brief builds the table that makes findMoleculeByGlobalAtomIndex a
     * binary search instead of a walk over all molecules
     *
     * @details Reading a topology file looks up up to four atoms per line, so
     * with the walk its cost grew with atoms times molecules. The table is not
     * kept up to date: build it when the molecules are final and clear it
     * afterwards. A table that does not fit the number of molecules is
     * ignored.
     */
    void SimulationBox::buildAtomIndexLookup()
    {
        _atomIndexOffsets.clear();
        _atomIndexOffsets.reserve(_molecules.size() + 1);
        _atomIndexOffsets.push_back(0);

        for (const auto& molecule : _molecules)
            _atomIndexOffsets.push_back(
                _atomIndexOffsets.back() + molecule.getNumberOfAtoms()
            );
    }

    /**
     * @brief drops the table of buildAtomIndexLookup
     */
    void SimulationBox::clearAtomIndexLookup() { _atomIndexOffsets.clear(); }

    /**
     * @brief find molecule by atom index
     *
     * @details return a pair of a pointer to the molecule and the index of the
     * atom in the molecule. The global atom index starts at 1. With the table
     * of buildAtomIndexLookup this is a binary search, otherwise a walk over
     * the molecules; an index that is not found (0 or too large) is always
     * reported by the walk.
     *
     * @param atomIndex
     * @return pair<Molecule *, AtomIndex>
     */
    std::pair<Molecule*, AtomIndex> SimulationBox::
        findMoleculeByGlobalAtomIndex(const size_t atomIndex)
    {
        if (_atomIndexOffsets.size() == _molecules.size() + 1 &&
            atomIndex > 0 && atomIndex <= _atomIndexOffsets.back())
        {
            // first molecule whose cumulative atom count reaches the index
            const auto next = std::ranges::lower_bound(
                _atomIndexOffsets.begin() + 1,
                _atomIndexOffsets.end(),
                atomIndex
            );
            const auto moleculeIndex = static_cast<size_t>(
                std::distance(_atomIndexOffsets.begin() + 1, next)
            );
            const auto index = atomIndex - _atomIndexOffsets[moleculeIndex] - 1;

            return std::make_pair(&_molecules[moleculeIndex], AtomIndex{index});
        }

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
                return std::make_pair(&molecule, AtomIndex{index});
            }
        }

        throw exc::UserInputException(
            std::format(
                "Atom index {} out of range - total number of atoms: {}",
                atomIndex,
                sum
            )
        );
    }

    /**
     * @brief find necessary molecule types
     *
     * @details The user can specify more molecule types in the moldescriptor
     * file than actually necessary in the simulation. This function returns
     * only the molecule types which are also present in the simulation box in
     * _molecules.
     *
     * @return std::vector<Molecule>
     */
    std::vector<MoleculeType> SimulationBox::findNecessaryMoleculeTypes()
    {
        std::vector<MoleculeType> neededMolTypes;

        auto searchMoleculeTypes = [&neededMolTypes, this](const auto& molecule)
        {
            auto predicate = [&molecule](const auto& moleculeType)
            { return molecule.getMoltype() == moleculeType.getMoltype(); };

            const auto molType =
                std::ranges::find_if(neededMolTypes, predicate);

            if (molType == neededMolTypes.end() &&
                molecule.getMoltype() != MolType{0})
                neededMolTypes.push_back(findMoleculeType(molecule.getMoltype())
                );
        };

        std::ranges::for_each(_molecules, searchMoleculeTypes);

        return neededMolTypes;
    }

    /**
     * @brief set partial charges of molecules from molecule types
     *
     * @throw exc::UserInputException if molecule type not found in
     * _moleculeTypes
     *
     */
    void SimulationBox::setPartialChargesOfMoleculesFromMoleculeTypes()
    {
        auto setPartialCharges =
            [&moleculeTypes = _moleculeTypes](Molecule& molecule)
        {
            auto predicate = [&molecule](const auto& moleculeType)
            { return molecule.getMoltype() == moleculeType.getMoltype(); };

            const auto molType = std::ranges::find_if(moleculeTypes, predicate);

            if (molType != moleculeTypes.end())
                molecule.setPartialCharges(molType->getPartialCharges());

            else if (molecule.getMoltype() != MolType{0})
            {
                throw exc::UserInputException(
                    std::format(
                        "Molecule type {} not found in molecule types",
                        molecule.getMoltype().toString()
                    )
                );
            }
        };

        std::ranges::for_each(_molecules, setPartialCharges);
    }

    /**
     * @brief make external to internal global vdw types map
     *
     * @details the function consists of multiple steps:
     * 1) fill the external global vdw types vector with all external global vdw
     * types from all molecules 2) sort and erase duplicates 3) fill the
     * external to internal global vdw types map - internal vdw types are
     * defined via increasing indices 4) set the internal global vdw types for
     * all molecules
     *
     */
    void SimulationBox::setupExternalToInternalGlobalVdwTypesMap()
    {
        /****************************************************************************
         * 1) fill the external global vdw types vector with all external global
         *vdw types from all molecules
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
            for (AtomIndex i{0}; i.get() < molecule.getNumberOfAtoms(); ++i)
            {
                auto& atom = molecule.getAtom(i);

                const auto extType = atom.getExternalGlobalVDWType();

                atom.setInternalGlobalVDWType(extToIntGlobalVDWTypes.at(extType)
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

        _degreesOfFreedom =
            (3 * nAtoms) - settings::GeneralSettings::getDimensionality();
    }

    /**
     * @brief calculate total mass of simulationBox
     *
     */
    void SimulationBox::calculateTotalMass()
    {
        _totalMass = Mass::zero();

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
        PosMass3D comHelper = PosMass::zero();

        auto accumulateMassWeightedPos = [&comHelper](const auto& atom)
        { comHelper += atom->getMass() * atom->getPosition(); };

        std::ranges::for_each(_atoms, accumulateMassWeightedPos);

        _centerOfMass = comHelper / _totalMass;
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
     * @return Momentum3D
     */
    Momentum3D SimulationBox::calculateMomentum()
    {
        Momentum3D momentum = Momentum::zero();

        auto accumulateAtomicMomentum = [&momentum](const auto& atom)
        { momentum += atom->getMass() * atom->getVelocity(); };

        std::ranges::for_each(_atoms, accumulateAtomicMomentum);

        return momentum;
    }

    /**
     * @brief calculate angular momentum of simulationBox
     *
     */
    AngularMomentum3D SimulationBox::calculateAngularMomentum(
        const Momentum3D& momentum
    )
    {
        AngularMomentum3D angularMom = AngularMomentum::zero();

        calculateCenterOfMass();

        auto accumulateAngularMomentum = [&angularMom](const auto& atom)
        {
            const auto mass = atom->getMass();
            angularMom +=
                mass * cross(atom->getPosition(), atom->getVelocity());
        };

        std::ranges::for_each(_atoms, accumulateAngularMomentum);

        angularMom -= cross(_centerOfMass, momentum / _totalMass) * _totalMass;

        return angularMom;
    }

    /**
     * @brief calculate total force of simulationBox as scalar
     *
     * @return Force
     */
    Force SimulationBox::calculateTotalForce() const
    {
        const auto totalForce = calculateTotalForceVector();

        return norm(totalForce);
    }

    /**
     * @brief calculate total force of simulationBox as vector
     *
     * @return Force3D
     */
    Force3D SimulationBox::calculateTotalForceVector() const
    {
        Force3D totalForce = Force::zero();

        auto accumulateForce = [&totalForce](const auto& atom)
        { totalForce += atom->getForce(); };

        std::ranges::for_each(_atoms, accumulateForce);

        return totalForce;
    }

    /**
     * @brief calculate RMS force of simulationBox
     *
     * @return Force
     */
    Force SimulationBox::calculateRMSForce() const
    {
        const auto scalarForces = getAtomicScalarForces();

        return stl::rms(scalarForces);
    }

    /**
     * @brief calculate max scalar force of simulationBox
     *
     * @return Force
     */
    Force SimulationBox::calculateMaxForce() const
    {
        const auto scalarForces = getAtomicScalarForces();

        return stl::max(scalarForces);
    }

    /**
     * @brief calculate rms old force of simulationBox
     *
     * @return Force
     */
    Force SimulationBox::calculateRMSForceOld() const
    {
        const auto scalarForces = getAtomicScalarForcesOld();

        return stl::rms(scalarForces);
    }

    /**
     * @brief calculate max force old of simulationBox
     *
     * @return Force
     */
    Force SimulationBox::calculateMaxForceOld() const
    {
        const auto scalarForces = getAtomicScalarForcesOld();

        return stl::max(scalarForces);
    }

    /**
     * @brief calculate temperature of simulationBox
     *
     */
    Temperature SimulationBox::calculateTemperature()
    {
        MassVelSquared twoEkinSum = MassVelSquared::zero();

        auto accumulateTemperature = [&twoEkinSum](const auto& atom)
        { twoEkinSum += atom->getMass() * normSquared(atom->getVelocity()); };

        std::ranges::for_each(_atoms, accumulateTemperature);

        const auto temperature =
            twoEkinSum / static_cast<double>(_degreesOfFreedom) / gas_constant;

        return Temperature(temperature);
    }

    /**
     * @brief checks if the coulomb radius cut off is smaller than half of the
     * minimal box dimension
     *
     * @throw exc::UserInputException if coulomb radius cut off is larger than
     * half of the minimal box dimension
     */
    void SimulationBox::checkCoulRadiusCutOff(
        const ExceptionType& exceptionType,
        const Distance&      coulombRadiusCutOff
    ) const
    {
        if (getMinimalBoxDimension() < 2.0 * coulombRadiusCutOff)
        {
            const std::string message =
                "Coulomb radius cut off is larger than half of the minimal box "
                "dimension";

            if (exceptionType == ExceptionType::ManostatError)
                throw exc::ManostatException(message);

            throw exc::UserInputException(message);
        }
    }

    /**
     * @brief calculate density of simulationBox
     *
     */
    void SimulationBox::calculateDensity()
    {
        const auto volume = _box->getVolume();
        _density          = _totalMass / volume / avogadro_constant;
    }

    /**
     * @brief calculate box dimensions from density
     *
     */
    Pos3D SimulationBox::calcBoxDimFromDensity() const
    {
        return OrthorhombicBox::calcBoxDimFromDensity(_totalMass, _density);
    }

    /**
     * @brief calculate shift vector for a given position
     *
     */
    Pos3D SimulationBox::calcShiftVector(const Pos3D& position) const
    {
        return _box->calcShiftVector(position);
    }

    int SimulationBox::calcActiveMolCharge() const
    {
        int charge = 0;

        for (const auto& mol : getActiveMolecules()) charge += mol.getCharge();

        return charge;
    }

    /**
     * @brief initialize positions of all atoms
     *
     */
    void SimulationBox::initPositions(const double displacement)
    {
        rng::RandomNumberGenerator rng{};

        auto displacePositions = [&rng, displacement, this](auto& atom)
        {
            Pos3D position = atom->getPosition();

            const auto random =
                linalg::Vec3D{
                    rng.getUniformRealDistribution(-displacement, displacement),
                    rng.getUniformRealDistribution(-displacement, displacement),
                    rng.getUniformRealDistribution(-displacement, displacement)
                } *
                Pos3D::value_type::unit;

            position += random;

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
     * @brief reset all forces of all atoms, i.e. forces, inner forces and outer
     * forces
     *
     */
    void SimulationBox::resetAllForces()
    {
        auto resetForces = [](const auto& atom)
        {
            atom->setForceToZero();
            atom->setInnerForceToZero();
            atom->setOuterForceToZero();
        };

        std::ranges::for_each(_atoms, resetForces);
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

    /**
     * @brief reset inner forces of all atoms
     *
     */
    void SimulationBox::resetForcesInner()
    {
        auto resetForces = [](const auto& atom)
        { atom->setInnerForceToZero(); };

        std::ranges::for_each(_atoms, resetForces);
    }

    /**
     * @brief reset outer forces of all atoms
     *
     */
    void SimulationBox::resetForcesOuter()
    {
        auto resetForces = [](const auto& atom)
        { atom->setOuterForceToZero(); };

        std::ranges::for_each(_atoms, resetForces);
    }

    /**
     * @brief reset qmCharges of all atoms
     *
     */
    void SimulationBox::resetQMCharges()
    {
        auto reset = [](const auto& atom) { atom->resetQMCharge(); };

        std::ranges::for_each(_atoms, reset);
    }

    /**
     * @brief Remove net force from the QM atoms
     *
     * @details Computes the total force on the QM atoms, distributes the
     * opposite mean force equally to the QM atoms, and thereby enforces zero
     * total QM force. MM atoms are not touched.
     */
    void SimulationBox::removeNetForce()
    {
        const auto nAtoms = getNumberOfQMAtoms();

        if (nAtoms == 0)
            return;

        Force3D totalForce = Force::zero();

        for (const auto& atom : getQMAtoms()) totalForce += atom->getForce();

        const auto correctionForce = -totalForce / static_cast<double>(nAtoms);

        for (auto& atom : getQMAtoms()) atom->addForce(correctionForce);
    }

    /**
     * @brief flattens positions of each atom into a single vector of Pos3D
     *
     * @return std::vector<Pos3D>
     */
    std::vector<Pos3D> SimulationBox::flattenPositions() const
    {
        std::vector<Pos3D> positions;

        auto addPositions = [&positions](auto& atom)
        {
            const auto pos = atom->getPosition();

            positions.push_back(pos);
        };

        std::ranges::for_each(_atoms, addPositions);

        return positions;
    }

    /**
     * @brief flattens forces of each atom into a single vector of doubles
     *
     * @return std::vector<Force>
     */
    std::vector<Force> SimulationBox::flattenForces() const
    {
        std::vector<Force> forces;

        auto addForces = [&forces](auto& atom)
        {
            const auto force = atom->getForce();

            forces.push_back(force[0]);
            forces.push_back(force[1]);
            forces.push_back(force[2]);
        };

        std::ranges::for_each(_atoms, addForces);

        return forces;
    }

}   // namespace molsys
