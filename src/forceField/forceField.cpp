#include "forceField.hpp"

#include "exceptions.hpp"

#include <algorithm>
#include <format>       // for format
#include <functional>   // for identity
#include <ranges>       // for find_if, ranges::find_if
#include <string>       // for string

using namespace forceField;
using namespace std;

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
    auto isBondId = [id](const BondType &bondType) { return bondType.getId() == id; };

    if (const auto bondType = ranges::find_if(_bondTypes, isBondId); bondType != _bondTypes.end())
        return *bondType;
    else
        throw customException::TopologyException(format("Bond type with id {} not found.", id));
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
    auto isAngleId = [id](const AngleType &angleType) { return angleType.getId() == id; };

    if (const auto angleType = ranges::find_if(_angleTypes, isAngleId); angleType != _angleTypes.end())
        return *angleType;
    else
        throw customException::TopologyException(format("Angle type with id {} not found.", id));
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
    auto isDihedralId = [id](const DihedralType &dihedralType) { return dihedralType.getId() == id; };

    if (const auto dihedralType = ranges::find_if(_dihedralTypes, isDihedralId); dihedralType != _dihedralTypes.end())
        return *dihedralType;
    else
        throw customException::TopologyException(format("Dihedral type with id {} not found.", id));
}

/**
 * @brief find improper dihedral type by id
 *
 * @param id
 * @return const DihedralType&
 *
 * @throws customException::TopologyException if improper dihedral type with id not found
 */
const DihedralType &ForceField::findImproperDihedralTypeById(const size_t id) const
{
    auto isImproperDihedralId = [id](const DihedralType &dihedralType) { return dihedralType.getId() == id; };

    if (const auto dihedralType = ranges::find_if(_improperDihedralTypes, isImproperDihedralId);
        dihedralType != _improperDihedralTypes.end())
        return *dihedralType;
    else
        throw customException::TopologyException(format("Improper dihedral type with id {} not found.", id));
}

void ForceField::calculateBondedInteractions(const simulationBox::SimulationBox &box, physicalData::PhysicalData &physicalData)
{
    calculateBondInteractions(box, physicalData);
    calculateAngleInteractions(box, physicalData);
    calculateDihedralInteractions(box, physicalData);
    calculateImproperDihedralInteractions(box, physicalData);
}

void ForceField::calculateBondInteractions(const simulationBox::SimulationBox &box, physicalData::PhysicalData &physicalData)
{
    ranges::for_each(_bonds,
                     [&box, &physicalData, this](auto &bond)
                     { bond.calculateEnergyAndForces(box, physicalData, *_coulombPotential, *_nonCoulombPotential); });
}

void ForceField::calculateAngleInteractions(const simulationBox::SimulationBox &box, physicalData::PhysicalData &physicalData)
{
    ranges::for_each(_angles,
                     [&box, &physicalData, this](auto &angle)
                     { angle.calculateEnergyAndForces(box, physicalData, *_coulombPotential, *_nonCoulombPotential); });
}

void ForceField::calculateDihedralInteractions(const simulationBox::SimulationBox &box, physicalData::PhysicalData &physicalData)
{
    ranges::for_each(_dihedrals,
                     [&box, &physicalData, this](auto &dihedral)
                     { dihedral.calculateEnergyAndForces(box, physicalData, false, *_coulombPotential, *_nonCoulombPotential); });
}

void ForceField::calculateImproperDihedralInteractions(const simulationBox::SimulationBox &box,
                                                       physicalData::PhysicalData         &physicalData)
{
    ranges::for_each(
        _improperDihedrals,
        [&box, &physicalData, this](auto &improperDihedral)
        { improperDihedral.calculateEnergyAndForces(box, physicalData, true, *_coulombPotential, *_nonCoulombPotential); });
}