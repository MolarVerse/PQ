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
#include "typeAliases.hpp"

using namespace simulationBox;

/************************
 *                      *
 * standard add methods *
 *                      *
 ************************/

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
 * @brief Get the number of QM atoms
 *
 * @return size_t
 */
size_t SimulationBox::getNumberOfQMAtoms() const { return _qmAtoms.size(); }

/**
 * @brief get the density
 *
 * @return double
 */
double SimulationBox::getDensity() const { return _density; }

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
 * @brief get all QM atoms
 *
 * @return std::vector<std::shared_ptr<Atom>>&
 */
std::vector<std::shared_ptr<Atom>> &SimulationBox::getQMAtoms()
{
    return _qmAtoms;
}

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
 * @brief set the density
 *
 * @param centerOfMass
 */
void SimulationBox::setDensity(const double density) { _density = density; }

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