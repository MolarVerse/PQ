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

#include "forceFieldClass.hpp"

#include <algorithm>
#include <format>       // for format
#include <functional>   // for identity
#include <ranges>       // for find_if, std::ranges::find_if
#include <string>       // for string

#include "exceptions.hpp"

using namespace forceField;

/**
 * @brief find bond type by id
 *
 * @param id
 * @return const BondType&
 *
 * @throws customException::TopologyException if bond type with id not found
 */
const BondType &ForceField::findBondTypeById(const size_t id) const
{
    auto isBondId = [id](const BondType &bondType)
    { return bondType.getId() == id; };

    const auto bondType = std::ranges::find_if(_bondTypes, isBondId);

    if (bondType != _bondTypes.end())
        return *bondType;
    else
        throw customException::TopologyException(
            std::format("Bond type with id {} not found.", id)
        );
}

/**
 * @brief find angle type by id
 *
 * @param id
 * @return const AngleType&
 *
 * @throws customException::TopologyException if angle type with id not found
 */
const AngleType &ForceField::findAngleTypeById(const size_t id) const
{
    auto isAngleId = [id](const AngleType &angleType)
    { return angleType.getId() == id; };

    const auto angleType = std::ranges::find_if(_angleTypes, isAngleId);

    if (angleType != _angleTypes.end())
        return *angleType;
    else
        throw customException::TopologyException(
            std::format("Angle type with id {} not found.", id)
        );
}

/**
 * @brief find dihedral type by id
 *
 * @param id
 * @return const DihedralType&
 *
 * @throws customException::TopologyException if dihedral type with id not found
 */
const DihedralType &ForceField::findDihedralTypeById(const size_t id) const
{
    auto isDihedralId = [id](const DihedralType &dihedralType)
    { return dihedralType.getId() == id; };

    auto       dihedrals    = _dihedralTypes;
    const auto dihedralType = std::ranges::find_if(dihedrals, isDihedralId);

    if (dihedralType != dihedrals.end())
        return *dihedralType;
    else
        throw customException::TopologyException(
            std::format("Dihedral type with id {} not found.", id)
        );
}

/**
 * @brief find improper dihedral type by id
 *
 * @param id
 * @return const DihedralType&
 *
 * @throws customException::TopologyException if improper dihedral type with id
 * not found
 */
const DihedralType &ForceField::findImproperDihedralTypeById(const size_t id
) const
{
    auto isImproperId = [id](const DihedralType &dihedralType)
    { return dihedralType.getId() == id; };

    auto       impropers    = _improperDihedralTypes;
    const auto dihedralType = std::ranges::find_if(impropers, isImproperId);

    if (dihedralType != impropers.end())
        return *dihedralType;
    else
        throw customException::TopologyException(
            std::format("Improper dihedral type with id {} not found.", id)
        );
}

/**
 * @brief find j-coupling type by id
 *
 * @param id
 * @return const JCouplingType&
 *
 * @throws customException::TopologyException if j-coupling type with id not
 * found
 */
const JCouplingType &ForceField::findJCouplingTypeById(const size_t id) const
{
    auto isJCouplingId = [id](const JCouplingType &jCouplingType)
    { return jCouplingType.getId() == id; };

    auto       jCouplings    = _jCouplingTypes;
    const auto jCouplingType = std::ranges::find_if(jCouplings, isJCouplingId);

    if (jCouplingType != _jCouplingTypes.end())
        return *jCouplingType;
    else
        throw customException::TopologyException(
            std::format("J-coupling type with id {} not found.", id)
        );
}

/**
 * @brief calculates all bonded interactions for:
 * 1) bonds
 * 2) angles
 * 3) dihedrals
 * 4) improper dihedrals
 *
 * @param box
 * @param physicalData
 */
void ForceField::calculateBondedInteractions(
    const simulationBox::SimulationBox &box,
    physicalData::PhysicalData         &physicalData
)
{
    calculateBondInteractions(box, physicalData);
    calculateAngleInteractions(box, physicalData);
    calculateDihedralInteractions(box, physicalData);
    calculateImproperDihedralInteractions(box, physicalData);
}

/**
 * @brief calculates all bond interactions
 *
 * @param box
 * @param physicalData
 */
void ForceField::calculateBondInteractions(
    const simulationBox::SimulationBox &box,
    physicalData::PhysicalData         &physicalData
)
{
    auto calculateBondInteraction = [&box, &physicalData, this](auto &bond)
    {
        bond.calculateEnergyAndForces(
            box,
            physicalData,
            *_coulombPotential,
            *_nonCoulombPotential
        );
    };

    std::ranges::for_each(_bonds, calculateBondInteraction);
}

/**
 * @brief calculates all angle interactions
 *
 * @param box
 * @param physicalData
 */
void ForceField::calculateAngleInteractions(
    const simulationBox::SimulationBox &box,
    physicalData::PhysicalData         &physicalData
)
{
    auto calculateAngleInteraction = [&box, &physicalData, this](auto &angle)
    {
        angle.calculateEnergyAndForces(
            box,
            physicalData,
            *_coulombPotential,
            *_nonCoulombPotential
        );
    };

    std::ranges::for_each(_angles, calculateAngleInteraction);
}

/**
 * @brief calculates all dihedral interactions
 *
 * @details set parameter isImproperDihedral to false
 *
 * @param box
 * @param physicalData
 */
void ForceField::calculateDihedralInteractions(
    const simulationBox::SimulationBox &box,
    physicalData::PhysicalData         &physicalData
)
{
    auto calculateDihedralInteraction =
        [&box, &physicalData, this](auto &dihedral)
    {
        dihedral.calculateEnergyAndForces(
            box,
            physicalData,
            false,
            *_coulombPotential,
            *_nonCoulombPotential
        );
    };

    std::ranges::for_each(_dihedrals, calculateDihedralInteraction);
}

/**
 * @brief calculates all improper dihedral interactions
 *
 * @details set parameter isImproperDihedral to true
 *
 * @param box
 * @param physicalData
 */
void ForceField::calculateImproperDihedralInteractions(
    const simulationBox::SimulationBox &box,
    physicalData::PhysicalData         &physicalData
)
{
    auto calculateImproperDihedralInteraction =
        [&box, &physicalData, this](auto &dihedral)
    {
        dihedral.calculateEnergyAndForces(
            box,
            physicalData,
            true,
            *_coulombPotential,
            *_nonCoulombPotential
        );
    };

    std::ranges::for_each(
        _improperDihedrals,
        calculateImproperDihedralInteraction
    );
}

/**
 * @brief calculates all j-coupling interactions
 *
 * @param box
 * @param physicalData
 */
void ForceField::calculateJCouplingInteractions(
    const simulationBox::SimulationBox &box,
    physicalData::PhysicalData         &physicalData
)
{
    if (!_jCouplings.empty())
        throw customException::UserInputException(
            "JCoupling interactions are not implemented yet."
        );

    // auto calculateJCouplingInteraction =
    //     [&box, &physicalData, this](auto &jCoupling)
    // {
    //     jCoupling.calculateEnergyAndForces(
    //         box,
    //         physicalData,
    //         *_coulombPotential,
    //         *_nonCoulombPotential
    //     );
    // };

    // std::ranges::for_each(_jCouplings, calculateJCouplingInteraction);
}