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

#include "molecule.hpp"

#include <algorithm>    // for std::ranges::for_each
#include <functional>   // for identity, equal_to
#include <iterator>     // for _Size, size
#include <ranges>       // for subrange

#include "box.hpp"                // for Box
#include "manostatSettings.hpp"   // for ManostatSettings
#include "settings.hpp"           // for Settings
#include "vector3d.hpp"           // for Vec3D

using namespace simulationBox;
using namespace linearAlgebra;
using namespace settings;

/**
 * @brief Construct a new Molecule:: Molecule object
 *
 * @param name
 */
Molecule::Molecule(const std::string_view name) : _name(name) {}

/**
 * @brief Construct a new Molecule:: Molecule object
 *
 * @param moltype
 */
Molecule::Molecule(const size_t moltype) : _moltype(moltype) {}

/**
 * @brief finds number of different atom types in molecule
 *
 * @return int
 */
size_t Molecule::getNumberOfAtomTypes()
{
    std::vector<size_t> extAtomTypes;

    const auto fill                = std::back_inserter(extAtomTypes);
    auto       getExternalAtomType = [](auto atom)
    { return atom->getExternalAtomType(); };

    std::ranges::transform(_atoms, fill, getExternalAtomType);

    const auto nUnique = std::ranges::size(std::ranges::unique(extAtomTypes));

    return getNumberOfAtoms() - nUnique;
}

/**
 * @brief calculates the center of mass of the molecule
 *
 * @details distances are calculated relative to the first atom
 *
 * @param box
 */
void Molecule::calculateCenterOfMass(const Box &box)
{
    _centerOfMass            = {0.0, 0.0, 0.0};
    const auto positionAtom1 = _atoms[0]->getPosition();

    for (const auto &atom : _atoms)
    {
        const auto mass     = atom->getMass();
        const auto position = atom->getPosition();
        const auto deltaPos = position - positionAtom1;

        _centerOfMass += mass * (position - box.calcShiftVector(deltaPos));
    }

    _centerOfMass /= getMolMass();

    _centerOfMass -= box.calcShiftVector(_centerOfMass);
}

/**
 * @brief scales the positions of the molecule by shifting the center of mass
 *
 * @details scaling has to be done in orthogonal space since pressure scaling is
 * done in orthogonal space
 *
 * @param shiftFactors
 */
void Molecule::scale(const tensor3D &shiftTensor, const Box &box)
{
    auto centerOfMass = _centerOfMass;

    if (ManostatSettings::getIsotropy() != Isotropy::FULL_ANISOTROPIC)
        centerOfMass = box.toOrthoSpace(_centerOfMass);

    const auto shift = shiftTensor * centerOfMass - centerOfMass;

    auto scaleAtomPosition = [&box, shift](auto atom)
    {
        auto position = atom->getPosition();

        if (ManostatSettings::getIsotropy() != Isotropy::FULL_ANISOTROPIC)
            position = box.toOrthoSpace(position);

        position += shift;

        if (ManostatSettings::getIsotropy() != Isotropy::FULL_ANISOTROPIC)
            position = box.toSimSpace(position);

        atom->setPosition(position);
    };

    std::ranges::for_each(_atoms, scaleAtomPosition);
}

/**
 * @brief returns the external global vdw types of the atoms in the molecule
 *
 * @return std::vector<size_t>
 */
std::vector<size_t> Molecule::getExternalGlobalVDWTypes() const
{
    std::vector<size_t> externalGlobalVDWTypes(getNumberOfAtoms());

    for (size_t i = 0; i < getNumberOfAtoms(); ++i)
        externalGlobalVDWTypes[i] = _atoms[i]->getExternalGlobalVDWType();

    return externalGlobalVDWTypes;
}

/**
 * @brief returns the atom masses of the atoms in the molecule
 *
 * @return std::vector<double>
 */
std::vector<double> Molecule::getAtomMasses() const
{
    std::vector<double> atomMasses(getNumberOfAtoms());

    for (size_t i = 0; i < getNumberOfAtoms(); ++i)
        atomMasses[i] = _atoms[i]->getMass();

    return atomMasses;
}

/**
 * @brief returns the partial charges of the atoms in the molecule
 *
 * @return std::vector<double>
 */
std::vector<double> Molecule::getPartialCharges() const
{
    std::vector<double> partialCharges(getNumberOfAtoms());

    for (size_t i = 0; i < getNumberOfAtoms(); ++i)
        partialCharges[i] = _atoms[i]->getPartialCharge();

    return partialCharges;
}

/**
 * @brief Determines if this molecule should be treated as a MM molecule
 *
 * @details The classification logic is as follows:
 * - For MM-only simulations: all molecules are MM molecules
 * - For QM-only simulations: no molecules are MM molecules
 * - For hybrid QM/MM simulations: active molecules are MM molecules
 *
 * @return true if the molecule should be treated with MM methods, false
 * otherwise
 */
bool Molecule::isMMMolecule() const
{
    if (Settings::isMMOnlyJobtype())
        return true;

    if (Settings::isQMOnlyJobtype())
        return false;

    if (isActive())
        return true;

    return false;
}

/**
 * @brief sets the partial charges of the atoms in the molecule
 *
 * @param partialCharges
 */
void Molecule::setPartialCharges(const std::vector<double> &partialCharges)
{
    for (size_t i = 0; i < getNumberOfAtoms(); ++i)
        _atoms[i]->setPartialCharge(partialCharges[i]);
}

/**
 * @brief sets the forces of the atoms in the molecule to zero
 *
 */
void Molecule::setAtomForcesToZero()
{
    std::ranges::for_each(_atoms, [](auto atom) { atom->setForceToZero(); });
}

/**
 * @brief activates the molecule and it's atoms for hybrid calculations
 *
 */
void Molecule::activateMolecule()
{
    _isActive = true;
    for (auto &atom : getAtoms()) atom->setActive(true);
}

/**
 * @brief deactivates the molecule and it's atoms for hybrid calculations
 *
 */
void Molecule::deactivateMolecule()
{
    _isActive = false;
    for (auto &atom : getAtoms()) atom->setActive(false);
}

/****************************************
 *                                      *
 * standard adder methods for atom data *
 *                                      *
 *****************************************/

/**
 * @brief adds an atom to the molecule
 *
 * @param atom
 */
void Molecule::addAtom(const std::shared_ptr<Atom> atom)
{
    _atoms.push_back(atom);
}

/**
 * @brief add a Vec3D to the current position of the atom by index
 *
 * @param index
 * @param position
 */
void Molecule::addAtomPosition(const size_t index, const Vec3D &position)
{
    _atoms[index]->addPosition(position);
}

/**
 * @brief  add a Vec3D to the current velocity of the atom by index
 *
 * @param index
 * @param velocity
 */
void Molecule::addAtomVelocity(const size_t index, const Vec3D &velocity)
{
    _atoms[index]->addVelocity(velocity);
}

/**
 * @brief add a Vec3D to the current force of the atom by index
 *
 * @param index
 * @param force
 */
void Molecule::addAtomForce(const size_t index, const Vec3D &force)
{
    _atoms[index]->addForce(force);
}

/**
 * @brief add a Vec3D to the current shift force of the atom by index
 *
 * @param index
 * @param shiftForce
 */
void Molecule::addAtomShiftForce(const size_t index, const Vec3D &shiftForce)
{
    _atoms[index]->addShiftForce(shiftForce);
}

/*****************************************
 *                                       *
 * standard setter methods for atom data *
 *                                       *
 ****************************************/

/**
 * @brief set the position of the atom by index
 *
 * @param index
 * @param position
 */
void Molecule::setAtomPosition(const size_t index, const Vec3D &position)
{
    _atoms[index]->setPosition(position);
}

/**
 * @brief set the velocity of the atom by index
 *
 * @param index
 * @param velocity
 */
void Molecule::setAtomVelocity(const size_t index, const Vec3D &velocity)
{
    _atoms[index]->setVelocity(velocity);
}

/**
 * @brief set the force of the atom by index
 *
 * @param index
 * @param force
 */
void Molecule::setAtomForce(const size_t index, const Vec3D &force)
{
    _atoms[index]->setForce(force);
}

/**
 * @brief set the shift force of the atom by index
 *
 * @param index
 * @param shiftForce
 */
void Molecule::setAtomShiftForce(const size_t index, const Vec3D &shiftForce)
{
    _atoms[index]->setShiftForce(shiftForce);
}

/****************************************
 *                                      *
 * standard getters for atom properties *
 *                                      *
 *****************************************/

/**
 * @brief returns the position of the atom by index
 *
 * @param index
 * @return Vec3D
 */
Vec3D Molecule::getAtomPosition(const size_t index) const
{
    return _atoms[index]->getPosition();
}

/**
 * @brief returns the positions of all atoms
 *
 * @return std::vector<Vec3D>
 */
std::vector<Vec3D> Molecule::getAtomPositions() const
{
    std::vector<Vec3D> positions;
    for (const auto &atom : _atoms) positions.push_back(atom->getPosition());

    return positions;
}

/**
 * @brief returns the velocity of the atom by index
 *
 * @param index
 * @return Vec3D
 */
Vec3D Molecule::getAtomVelocity(const size_t index) const
{
    return _atoms[index]->getVelocity();
}

/**
 * @brief returns the force of the atom by index
 *
 * @param index
 * @return Vec3D
 */
Vec3D Molecule::getAtomForce(const size_t index) const
{
    return _atoms[index]->getForce();
}

/**
 * @brief returns the shift force of the atom by index
 *
 * @param index
 * @return Vec3D
 */
Vec3D Molecule::getAtomShiftForce(const size_t index) const
{
    return _atoms[index]->getShiftForce();
}

/**
 * @brief returns the atomic number of the atom by index
 *
 * @param index
 * @return int
 */
int Molecule::getAtomicNumber(const size_t index) const
{
    return _atoms[index]->getAtomicNumber();
}

/**
 * @brief returns the mass of the atom by index
 *
 * @param index
 * @return double
 */
double Molecule::getAtomMass(const size_t index) const
{
    return _atoms[index]->getMass();
}

/**
 * @brief returns the partial charge of the atom by index
 *
 * @param index
 * @return double
 */
double Molecule::getPartialCharge(const size_t index) const
{
    return _atoms[index]->getPartialCharge();
}

/**
 * @brief returns the atom type of the atom by index
 *
 * @param index
 * @return size_t
 */
size_t Molecule::getAtomType(const size_t index) const
{
    return _atoms[index]->getAtomType();
}

/**
 * @brief returns the internal global vdw type of the atom by index
 *
 * @param index
 * @return size_t
 */
size_t Molecule::getInternalGlobalVDWType(const size_t index) const
{
    return _atoms[index]->getInternalGlobalVDWType();
}

/**
 * @brief returns the atom name of the atom by index
 *
 * @param index
 * @return std::string
 */
std::string Molecule::getAtomName(const size_t index) const
{
    return _atoms[index]->getName();
}

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

/**
 * @brief returns the moltype of the molecule
 *
 * @return size_t
 */
size_t Molecule::getMoltype() const { return _moltype; }

/**
 * @brief returns the number of atoms in the molecule
 *
 * @return size_t
 */
size_t Molecule::getNumberOfAtoms() const { return _numberOfAtoms; }

/**
 * @brief returns the number of degrees of freedom of the molecule
 *
 * @return size_t
 */
size_t Molecule::getDegreesOfFreedom() const { return 3 * getNumberOfAtoms(); }

/**
 * @brief returns the charge of the molecule
 *
 * @return int
 */

int Molecule::getCharge() const { return _charge; }

/**
 * @brief returns the molecular mass of the molecule
 *
 * @return double
 */
double Molecule::getMolMass() const { return _molMass; }

/**
 * @brief returns the name of the molecule
 *
 * @return std::string
 */
std::string Molecule::getName() const { return _name; }

/**
 * @brief returns the center of mass of the molecule
 *
 * @return Vec3D
 */
Vec3D Molecule::getCenterOfMass() const { return _centerOfMass; }

/**
 * @brief return the Hybrid zone of the molecule
 *
 * @return HybridZone
 */
HybridZone Molecule::getHybridZone() const { return _hybridZone; }

/**
 * @brief return if the molecule is activate for hybrid calculations
 *
 * @return isActive
 */
bool Molecule::isActive() const { return _isActive; }

/**
 * @brief return the smoothing factor of the molecule for hybrid calculations
 *
 * @return double
 */
double Molecule::getSmoothingFactor() const { return _smoothingFactor; }

/**
 * @brief returns the atom by index
 *
 * @param index
 * @return Atom
 */
Atom &Molecule::getAtom(const size_t index) { return *(_atoms[index]); }

/**
 * @brief returns the atoms of the molecule
 *
 * @return std::vector<Atom>
 */
std::vector<std::shared_ptr<Atom>> &Molecule::getAtoms() { return _atoms; }

/**
 * @brief returns the atoms of the molecule
 *
 * @return std::vector<Atom>
 */
const std::vector<std::shared_ptr<Atom>> &Molecule::getAtoms() const
{
    return _atoms;
}

/**
 * @brief return if the molecule is forced to be in the inner region for hybrid
 * calculations
 *
 * @return true
 * @return false
 */
bool Molecule::isForcedInner() const { return _isForcedInner; }

/**
 * @brief return if the molecule is forced to be in the outer region for hybrid
 * calculations
 *
 * @return true
 * @return false
 */
bool Molecule::isForcedOuter() const { return _isForcedOuter; }

/***************************
 *                         *
 * standard setter methods *
 *                         *
 ***************************/

/**
 * @brief set the name of the molecule
 *
 * @param name
 */
void Molecule::setName(const std::string_view name) { _name = name; }

/**
 * @brief set the number of atoms in the molecule
 *
 * @param numberOfAtoms
 */
void Molecule::setNumberOfAtoms(const size_t numberOfAtoms)
{
    _numberOfAtoms = numberOfAtoms;
}

/**
 * @brief set the moltype of the molecule
 *
 * @param moltype
 */
void Molecule::setMoltype(const size_t moltype) { _moltype = moltype; }

/**
 * @brief set the charge of the molecule
 *
 * @param charge
 */
void Molecule::setCharge(const int charge) { _charge = charge; }

/**
 * @brief set the molecular mass of the molecule
 *
 * @param molMass
 */
void Molecule::setMolMass(const double molMass) { _molMass = molMass; }

/**
 * @brief set the center of mass of the molecule
 *
 * @param centerOfMass
 */
void Molecule::setCenterOfMass(const Vec3D &centerOfMass)
{
    _centerOfMass = centerOfMass;
}

/**
 * @brief set the Hybrid zone of the molecule
 *
 * @param hybridZone
 */
void Molecule::setHybridZone(const HybridZone hybridZone)
{
    _hybridZone = hybridZone;
}

/**
 * @brief set the smoothing factor of the molecule for hybrid calculations
 *
 * @param factor
 */
void Molecule::setSmoothingFactor(const double factor)
{
    _smoothingFactor = factor;
}

/**
 * @brief set if the molecule is forced to be in the inner region for hybrid
 * calculations
 *
 * @param isForcedInner
 */
void Molecule::setForcedInner(const bool isForcedInner)
{
    _isForcedInner = isForcedInner;
}

/**
 * @brief set if the molecule is forced to be in the outer region for hybrid
 * calculations
 *
 * @param isForcedOuter
 */
void Molecule::setForcedOuter(const bool isForcedOuter)
{
    _isForcedOuter = isForcedOuter;
}