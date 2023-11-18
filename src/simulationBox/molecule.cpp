/*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
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

#include "box.hpp"                // for Box
#include "manostatSettings.hpp"   // for ManostatSettings
#include "vector3d.hpp"           // for Vec3D

#include <algorithm>    // for std::ranges::for_each
#include <functional>   // for identity, equal_to
#include <iterator>     // for _Size, size
#include <ranges>       // for subrange

using namespace simulationBox;

/**
 * @brief finds number of different atom types in molecule
 *
 * @return int
 */
size_t Molecule::getNumberOfAtomTypes()
{
    std::vector<size_t> externalAtomTypes;

    std::ranges::transform(_atoms, std::back_inserter(externalAtomTypes), [](auto atom) { return atom->getExternalAtomType(); });

    return getNumberOfAtoms() - std::ranges::size(std::ranges::unique(externalAtomTypes));
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

        _centerOfMass += mass * (position - box.calculateShiftVector(position - positionAtom1));
    }

    _centerOfMass /= getMolMass();

    _centerOfMass -= box.calculateShiftVector(_centerOfMass);
}

/**
 * @brief scales the positions of the molecule by shifting the center of mass
 *
 * @details scaling has to be done in orthogonal space since pressure scaling is done in orthogonal space
 *
 * @param shiftFactors
 */
void Molecule::scale(const linearAlgebra::tensor3D &shiftTensor, const Box &box)
{
    auto centerOfMass = _centerOfMass;

    if (settings::ManostatSettings::getIsotropy() != settings::Isotropy::FULL_ANISOTROPIC)
        centerOfMass = box.transformIntoOrthogonalSpace(_centerOfMass);

    const auto shift = shiftTensor * centerOfMass - centerOfMass;

    auto scaleAtomPosition = [&box, shift](auto atom)
    {
        auto position = atom->getPosition();
        if (settings::ManostatSettings::getIsotropy() != settings::Isotropy::FULL_ANISOTROPIC)
            position = box.transformIntoOrthogonalSpace(position);

        position += shift;

        if (settings::ManostatSettings::getIsotropy() != settings::Isotropy::FULL_ANISOTROPIC)
            position = box.transformIntoSimulationSpace(position);

        atom->setPosition(position);
    };

    std::ranges::for_each(_atoms, scaleAtomPosition);
}
// void Molecule::scale(const linearAlgebra::tensor3D &shiftFactors, const Box &box)
// {
//     const auto centerOfMass = box.transformIntoOrthogonalSpace(_centerOfMass);

//     const auto shift = centerOfMass * (shiftFactors - 1.0);

//     auto scaleAtomPosition = [&box, shift](auto atom)
//     {
//         auto position = atom->getPosition();
//         position      = box.transformIntoOrthogonalSpace(position);

//         position += shift;

//         position = box.transformIntoSimulationSpace(position);

//         atom->setPosition(position);
//     };

//     std::ranges::for_each(_atoms, scaleAtomPosition);
// }

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