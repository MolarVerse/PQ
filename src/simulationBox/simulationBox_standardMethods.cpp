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

#include <ranges>   // for views::filter

#include "settings.hpp"   // for Settings::getJobtype, JobType
#include "simulationBox.hpp"

using namespace simulationBox;

/************************
 *                      *
 * standard add methods *
 *                      *
 ************************/

/**
 * @brief Add an atom to the simulation box
 *
 * @param atom
 */
void SimulationBox::addAtom(const std::shared_ptr<Atom> atom)
{
    _atoms.push_back(atom);
}

/**
 * @brief Add a QM atom to the simulation box
 *
 * @param atom
 */
void SimulationBox::addQMAtom(const std::shared_ptr<Atom> atom)
{
    _qmAtoms.push_back(atom);
}

/**
 * @brief Add a molecule to the simulation box
 *
 * @param molecule
 */
void SimulationBox::addMolecule(const Molecule &molecule)
{
    _molecules.push_back(molecule);
}

/**
 * @brief Add a molecule type to the simulation box
 *
 * @param molecule
 */
void SimulationBox::addMoleculeType(const MoleculeType &molecule)
{
    _moleculeTypes.push_back(molecule);
}

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

/**
 * @brief Get the water type
 *
 * @return int
 */
int SimulationBox::getWaterType() const { return _waterType; }

/**
 * @brief Get the ammonia type
 *
 * @return int
 */
int SimulationBox::getAmmoniaType() const { return _ammoniaType; }

/**
 * @brief Get the number of molecules
 *
 * @return size_t
 */
size_t SimulationBox::getNumberOfMolecules() const { return _molecules.size(); }

/**
 * @brief Get degrees of freedom
 *
 * @return size_t
 */
size_t SimulationBox::getDegreesOfFreedom() const { return _degreesOfFreedom; }

/**
 * @brief Get the number of atoms
 *
 * @return size_t
 */
size_t SimulationBox::getNumberOfAtoms() const { return _atoms.size(); }

/**
 * @brief Get the number of QM atoms
 *
 * @return size_t
 */
size_t SimulationBox::getNumberOfQMAtoms() const { return _qmAtoms.size(); }

/**
 * @brief get the total mass
 *
 * @return double
 */
double SimulationBox::getTotalMass() const { return _totalMass; }

/**
 * @brief get the total charge
 *
 * @return double
 */
double SimulationBox::getTotalCharge() const { return _totalCharge; }

/**
 * @brief get the density
 *
 * @return double
 */
double SimulationBox::getDensity() const { return _density; }

/**
 * @brief get the center of mass
 *
 * @return linearAlgebra::Vec3D&
 */
linearAlgebra::Vec3D &SimulationBox::getCenterOfMass() { return _centerOfMass; }

/**
 * @brief get atom by index
 *
 * @param index
 * @return Atom&
 */
Atom &SimulationBox::getAtom(const size_t index) { return *(_atoms[index]); }

/**
 * @brief get QM atom by index
 *
 * @param index
 * @return Atom&
 */
Atom &SimulationBox::getQMAtom(const size_t index)
{
    return *(_qmAtoms[index]);
}

/**
 * @brief get molecule by index
 *
 * @param index
 * @return Molecule&
 */
Molecule &SimulationBox::getMolecule(const size_t index)
{
    return _molecules[index];
}

/**
 * @brief get molecule type by index
 *
 * @param index
 * @return MoleculeType&
 */
MoleculeType &SimulationBox::getMoleculeType(const size_t index)
{
    return _moleculeTypes[index];
}

/**
 * @brief get atomic scalar forces
 *
 * @return std::vector<double>
 */
std::vector<double> SimulationBox::getAtomicScalarForces() const
{
    std::vector<double> atomicScalarForces;

    for (const auto &atom : _atoms)
        atomicScalarForces.push_back(norm(atom->getForce()));

    return atomicScalarForces;
}

/**
 * @brief get atomic scalar forces old
 *
 * @return std::vector<double>
 */
std::vector<double> SimulationBox::getAtomicScalarForcesOld() const
{
    std::vector<double> atomicScalarForces;

    for (const auto &atom : _atoms)
        atomicScalarForces.push_back(norm(atom->getForceOld()));

    return atomicScalarForces;
}

/**
 * @brief get all atoms
 *
 * @return std::vector<std::shared_ptr<Atom>>&
 */
std::vector<std::shared_ptr<Atom>> &SimulationBox::getAtoms() { return _atoms; }

/**
 * @brief get all QM atoms
 *
 * @return std::vector<std::shared_ptr<Atom>>&
 */
std::vector<std::shared_ptr<Atom>> &SimulationBox::getQMAtoms()
{
    return _qmAtoms;
}

/**
 * @brief get all QM atoms using range-based filtering
 *
 * @return a view/range of QM atoms filtered from all atoms
 *
 * @details This function returns a range-based view that filters atoms
 *          from _atoms based on job type and QMMM criteria. For QM_MD jobs,
 *          all atoms are considered QM atoms. For other job types, atoms
 *          are filtered based on their QMMMType.
 */
auto SimulationBox::getQMAtomsNew() const
{
    return _atoms | std::ranges::views::filter(
                        [](const auto &atom)
                        {
                            if (settings::Settings::getJobtype() ==
                                settings::JobType::QM_MD)
                                return true;

                            else if (atom->getQMMMType() == QMMMType::QM)
                                return true;

                            else
                                return atom->isQMOnly();
                        }
                    );
}

/**
 * @brief get all molecules
 *
 * @return std::vector<Molecule>&
 */
std::vector<Molecule> &SimulationBox::getMolecules() { return _molecules; }

/**
 * @brief get all molecule types
 *
 * @return std::vector<MoleculeType>&
 */
std::vector<MoleculeType> &SimulationBox::getMoleculeTypes()
{
    return _moleculeTypes;
}

/**
 * @brief get the external global VDW types
 *
 * @return std::vector<size_t>&
 */
std::vector<size_t> &SimulationBox::getExternalGlobalVdwTypes()
{
    return _externalGlobalVdwTypes;
}

/**
 * @brief get the external to internal global VDW types map
 *
 * @return std::unordered_map<size_t, size_t>&
 */
std::map<size_t, size_t> &SimulationBox::getExternalToInternalGlobalVDWTypes()
{
    return _externalToInternalGlobalVDWTypes;
}

/**
 * @brief get the Box/Cell
 *
 * @return Box&
 */
Box &SimulationBox::getBox() { return *_box; }

/**
 * @brief get the Box/Cell
 *
 * @return Box&
 */
Box &SimulationBox::getBox() const { return *_box; }

/**
 * @brief get the Box/Cell pointer
 *
 * @return std::shared_ptr<Box>
 */
std::shared_ptr<Box> SimulationBox::getBoxPtr() { return _box; }

/**
 * @brief get the Box/Cell pointer
 *
 * @return std::shared_ptr<Box>
 */
std::shared_ptr<Box> SimulationBox::getBoxPtr() const { return _box; }

/**
 * @brief get all positions of all atoms
 *
 * @return std::vector<linearAlgebra::Vec3D>
 */
std::vector<linearAlgebra::Vec3D> SimulationBox::getPositions() const
{
    std::vector<linearAlgebra::Vec3D> positions;

    for (const auto &atom : _atoms) positions.push_back(atom->getPosition());

    return positions;
}

/**
 * @brief get all velocities of all atoms
 *
 * @return std::vector<linearAlgebra::Vec3D>
 */
std::vector<linearAlgebra::Vec3D> SimulationBox::getVelocities() const
{
    std::vector<linearAlgebra::Vec3D> velocities;

    for (const auto &atom : _atoms) velocities.push_back(atom->getVelocity());

    return velocities;
}

/**
 * @brief get all forces of all atoms
 *
 * @return std::vector<linearAlgebra::Vec3D>
 */
std::vector<linearAlgebra::Vec3D> SimulationBox::getForces() const
{
    std::vector<linearAlgebra::Vec3D> forces;

    for (const auto &atom : _atoms) forces.push_back(atom->getForce());

    return forces;
}

/**
 * @brief get all atomic numbers of all atoms
 *
 * @return std::vector<int>
 */
std::vector<int> SimulationBox::getAtomicNumbers() const
{
    std::vector<int> atomicNumbers;

    for (const auto &atom : _atoms)
        atomicNumbers.push_back(atom->getAtomicNumber());

    return atomicNumbers;
}

/**
 * @brief flattens positions of each atom into a single vector of doubles
 *
 * @return std::vector<double>
 */
std::vector<double> SimulationBox::flattenPositions() const
{
    std::vector<double> positions;

    auto addPositions = [&positions](auto &atom)
    {
        const auto position = atom->getPosition();

        positions.push_back(position[0]);
        positions.push_back(position[1]);
        positions.push_back(position[2]);
    };

    std::ranges::for_each(_atoms, addPositions);

    return positions;
}

/***************************
 *                         *
 * standard setter methods *
 *                         *
 ***************************/

/**
 * @brief set the water type
 *
 * @param waterType
 */
void SimulationBox::setWaterType(const int waterType)
{
    _waterType = waterType;
}

/**
 * @brief set the ammonia type
 *
 * @param ammoniaType
 */
void SimulationBox::setAmmoniaType(const int ammoniaType)
{
    _ammoniaType = ammoniaType;
}

/**
 * @brief set the total mass
 *
 * @param centerOfMass
 */
void SimulationBox::setTotalMass(const double totalMass)
{
    _totalMass = totalMass;
}

/**
 * @brief set the total charge
 *
 * @param centerOfMass
 */
void SimulationBox::setTotalCharge(const double totalCharge)
{
    _totalCharge = totalCharge;
}

/**
 * @brief set the density
 *
 * @param centerOfMass
 */
void SimulationBox::setDensity(const double density) { _density = density; }

/**
 * @brief set the degrees of freedom
 *
 * @param centerOfMass
 */
void SimulationBox::setDegreesOfFreedom(const size_t degreesOfFreedom)
{
    _degreesOfFreedom = degreesOfFreedom;
}

/**********************************************
 *                                            *
 * Forwards the box methods to the box object *
 *                                            *
 **********************************************/

/**
 * @brief applies periodic boundary conditions to a position
 *
 * @param position
 */
void SimulationBox::applyPBC(linearAlgebra::Vec3D &position) const
{
    _box->applyPBC(position);
}

/**
 * @brief scales the box
 *
 * @param scalingTensor
 */
void SimulationBox::scaleBox(const linearAlgebra::tensor3D &scalingTensor)
{
    _box->scaleBox(scalingTensor);
    calculateDensity();
}

/**
 * @brief calculates the volume
 *
 * @return double
 */
double SimulationBox::calculateVolume() const
{
    return _box->calculateVolume();
}

/**
 * @brief gets the minimal box dimension
 *
 * @return double
 */
double SimulationBox::getMinimalBoxDimension() const
{
    return _box->getMinimalBoxDimension();
}

/**
 * @brief gets the volume
 *
 * @return double
 */
double SimulationBox::getVolume() const { return _box->getVolume(); }

/**
 * @brief gets if the box size has changed
 *
 * @return bool
 */
bool SimulationBox::getBoxSizeHasChanged() const
{
    return _box->getBoxSizeHasChanged();
}

/**
 * @brief gets the box dimensions
 *
 * @return linearAlgebra::Vec3D
 */
linearAlgebra::Vec3D SimulationBox::getBoxDimensions() const
{
    return _box->getBoxDimensions();
}

/**
 * @brief gets the box angles
 *
 * @return linearAlgebra::Vec3D
 */
linearAlgebra::Vec3D SimulationBox::getBoxAngles() const
{
    return _box->getBoxAngles();
}

/**
 * @brief sets the volume
 *
 * @param volume
 */
void SimulationBox::setVolume(const double volume) const
{
    _box->setVolume(volume);
}

/**
 * @brief sets the box dimensions
 *
 * @param boxDimensions
 */
void SimulationBox::setBoxDimensions(const linearAlgebra::Vec3D &boxDimensions
) const
{
    _box->setBoxDimensions(boxDimensions);
}

/**
 * @brief sets if the box size has changed
 *
 * @param boxSizeHasChanged
 */
void SimulationBox::setBoxSizeHasChanged(const bool boxSizeHasChanged) const
{
    _box->setBoxSizeHasChanged(boxSizeHasChanged);
}