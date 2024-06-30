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

#include "atom.hpp"

#include "atomMassMap.hpp"        // for atomMassMap
#include "box.hpp"                // for Box
#include "exceptions.hpp"         // for MolDescriptorException
#include "manostatSettings.hpp"   // for ManostatSettings
#include "stringUtilities.hpp"    // for toLowerCopy
#include "vector3d.hpp"           // for Vec3D

using simulationBox::Atom;

/**
 * @brief sets the mass of the atom
 *
 * @throw customException::MolDescriptorException if the atom name is invalid
 */
void Atom::initMass()
{
    const auto keyword = utilities::toLowerCopy(_name);
    if (!constants::atomMassMap.contains(keyword))
        throw customException::MolDescriptorException(
            "Invalid atom name \"" + keyword + "\""
        );
    else
        setMass(constants::atomMassMap.at(keyword));
}

/**
 * @brief updates the old position of the atom to the current position
 */
void Atom::updateOldPosition() { _positionOld = _position; }

/**
 * @brief updates the old velocity of the atom to the current velocity
 */
void Atom::updateOldVelocity() { _velocityOld = _velocity; }

/**
 * @brief updates the old force of the atom to the current force
 */
void Atom::updateOldForce() { _forceOld = _force; }

/*******************
 *                 *
 * scaling methods *
 *                 *
 *******************/

/**
 * @brief scales the velocity of the atom
 *
 * @param scaleFactor double
 */
void Atom::scaleVelocity(const double scaleFactor) { _velocity *= scaleFactor; }

/**
 * @brief scales the velocity of the atom by a Vec3D elementwise
 *
 * @param scaleFactor Vec3D
 */
void Atom::scaleVelocity(const linearAlgebra::Vec3D &scaleFactor)
{
    _velocity *= scaleFactor;
}

/**
 * @brief scales the velocities of the atom in orthogonal space
 *
 * @param scalingFactor
 * @param box
 */
void Atom::scaleVelocityOrthogonalSpace(
    const linearAlgebra::tensor3D &scalingTensor,
    const simulationBox::Box      &box
)
{
    if (settings::ManostatSettings::getIsotropy() !=
        settings::Isotropy::FULL_ANISOTROPIC)
        _velocity = box.toOrthoSpace(_velocity);

    _velocity = scalingTensor * _velocity;

    if (settings::ManostatSettings::getIsotropy() !=
        settings::Isotropy::FULL_ANISOTROPIC)
        _velocity = box.toSimSpace(_velocity);
}

/**************************
 *                        *
 * standard adder methods *
 *                        *
 **************************/

/**
 * @brief add a Vec3D to the current position of the atom
 *
 * @param position
 */
void Atom::addPosition(const linearAlgebra::Vec3D &position)
{
    _position += position;
}

/**
 * @brief  add a Vec3D to the current velocity of the atom
 *
 * @param velocity
 */
void Atom::addVelocity(const linearAlgebra::Vec3D &velocity)
{
    _velocity += velocity;
}

/**
 * @brief add a Vec3D to the current force of the atom
 *
 * @param force
 */
void Atom::addForce(const linearAlgebra::Vec3D &force) { _force += force; }

/**
 * @brief  add a force to the current force of the atom
 *
 * @param force_x
 * @param force_y
 * @param force_z
 */
void Atom::addForce(
    const double force_x,
    const double force_y,
    const double force_z
)
{
    _force += {force_x, force_y, force_z};
}

/**
 * @brief add a Vec3D to the current shift force of the atom
 *
 * @param shiftForce
 */
void Atom::addShiftForce(const linearAlgebra::Vec3D &shiftForce)
{
    _shiftForce += shiftForce;
}

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

/**
 * @brief return if the atom is QM only
 *
 * @return true
 * @return false
 */
bool Atom::isQMOnly() const { return _isQMOnly; }

/**
 * @brief return if the atom is MM only
 *
 * @return true
 * @return false
 */
bool Atom::isMMOnly() const { return _isMMOnly; }

/**
 * @brief return the name of the atom (element name)
 *
 * @return std::string
 */
std::string Atom::getName() const { return _name; }

/**
 * @brief return the atom type name
 *
 * @return std::string
 */
std::string Atom::getAtomTypeName() const { return _atomTypeName; }

/**
 * @brief return the external atom type
 *
 * @return size_t
 */
size_t Atom::getExternalAtomType() const { return _externalAtomType; }

/**
 * @brief return the atom type (internal)
 *
 * @return size_t
 */
size_t Atom::getAtomType() const { return _atomType; }

/**
 * @brief return the external global VDW type
 *
 * @return size_t
 */
size_t Atom::getExternalGlobalVDWType() const { return _externalGlobalVDWType; }

/**
 * @brief return the internal global VDW type
 *
 * @return size_t
 */
size_t Atom::getInternalGlobalVDWType() const { return _internalGlobalVDWType; }

/**
 * @brief return the atomic number of the atom
 *
 * @return int
 */
int Atom::getAtomicNumber() const { return _atomicNumber; }

/**
 * @brief return the mass of the atom
 *
 * @return double
 */
double Atom::getMass() const { return _mass; }

/**
 * @brief return the partial charge of the atom
 *
 * @return double
 */
double Atom::getPartialCharge() const { return _partialCharge; }

/**
 * @brief return the position of the atom
 *
 * @return linearAlgebra::Vec3D
 */
linearAlgebra::Vec3D Atom::getPosition() const { return _position; }

/**
 * @brief return the old position of the atom
 *
 * @return linearAlgebra::Vec3D
 */
linearAlgebra::Vec3D Atom::getPositionOld() const { return _positionOld; }

/**
 * @brief return the velocity of the atom
 *
 * @return linearAlgebra::Vec3D
 */
linearAlgebra::Vec3D Atom::getVelocity() const { return _velocity; }

/**
 * @brief return the force of the atom
 *
 * @return linearAlgebra::Vec3D
 */
linearAlgebra::Vec3D Atom::getForce() const { return _force; }

/**
 * @brief return the old force of the atom
 *
 * @return linearAlgebra::Vec3D
 */
linearAlgebra::Vec3D Atom::getForceOld() const { return _forceOld; }

/**
 * @brief return the shift force of the atom
 *
 * @return linearAlgebra::Vec3D
 */
linearAlgebra::Vec3D Atom::getShiftForce() const { return _shiftForce; }

/***************************
 *                         *
 * standard setter methods *
 *                         *
 ***************************/

/**
 * @brief set if the atom is QM only
 *
 * @param position
 */
void Atom::setQMOnly(const bool isQMOnly) { _isQMOnly = isQMOnly; }

/**
 * @brief set if the atom is MM only
 *
 * @param position
 */
void Atom::setMMOnly(const bool isMMOnly) { _isMMOnly = isMMOnly; }

/**
 * @brief set the name of the atom (element name)
 *
 * @param name
 */
void Atom::setName(const std::string_view &name) { _name = name; }

/**
 * @brief set the atom type name
 *
 * @param atomTypeName
 */
void Atom::setAtomTypeName(const std::string_view &atomTypeName)
{
    _atomTypeName = atomTypeName;
}

/**
 * @brief set the atomic number of the atom
 *
 * @param atomicNumber
 */
void Atom::setAtomicNumber(const int atomicNumber)
{
    _atomicNumber = atomicNumber;
}

/**
 * @brief set the mass of the atom
 *
 * @param mass
 */
void Atom::setMass(const double mass) { _mass = mass; }

/**
 * @brief set the partial charge of the atom
 *
 * @param partialCharge
 */
void Atom::setPartialCharge(const double partialCharge)
{
    _partialCharge = partialCharge;
}

/**
 * @brief set the atom type (internal)
 *
 * @param atomType
 */
void Atom::setAtomType(const size_t atomType) { _atomType = atomType; }

/**
 * @brief set the external atom type
 *
 * @param externalAtomType
 */
void Atom::setExternalAtomType(const size_t externalAtomType)
{
    _externalAtomType = externalAtomType;
}

/**
 * @brief set the external global VDW type
 *
 * @param externalGlobalVDWType
 */
void Atom::setExternalGlobalVDWType(const size_t externalGlobalVDWType)
{
    _externalGlobalVDWType = externalGlobalVDWType;
}

/**
 * @brief set the internal global VDW type
 *
 * @param internalGlobalVDWType
 */
void Atom::setInternalGlobalVDWType(const size_t internalGlobalVDWType)
{
    _internalGlobalVDWType = internalGlobalVDWType;
}

/**
 * @brief set the position of the atom
 *
 * @param position
 */
void Atom::setPosition(const linearAlgebra::Vec3D &position)
{
    _position = position;
}

/**
 * @brief set the velocity of the atom
 *
 * @param velocity
 */
void Atom::setVelocity(const linearAlgebra::Vec3D &velocity)
{
    _velocity = velocity;
}

/**
 * @brief set the force of the atom
 *
 * @param force
 */
void Atom::setForce(const linearAlgebra::Vec3D &force) { _force = force; }

/**
 * @brief set the shift force of the atom
 *
 * @param shiftForce
 */
void Atom::setShiftForce(const linearAlgebra::Vec3D &shiftForce)
{
    _shiftForce = shiftForce;
}

/**
 * @brief set the force of the atom to zero
 */
void Atom::setForceToZero() { _force = {0.0, 0.0, 0.0}; }

/**
 * @brief set the old position of the atom
 *
 * @param position
 */
void Atom::setPositionOld(const linearAlgebra::Vec3D &position)
{
    _positionOld = position;
}

/**
 * @brief set the old velocity of the atom
 *
 * @param velocity
 */
void Atom::setVelocityOld(const linearAlgebra::Vec3D &velocity)
{
    _velocityOld = velocity;
}

/**
 * @brief set the old force of the atom
 *
 * @param force
 */
void Atom::setForceOld(const linearAlgebra::Vec3D &force) { _forceOld = force; }