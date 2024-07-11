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
using namespace customException;
using namespace simulationBox;
using namespace physicalData;
using namespace potential;

/**
 * @brief clones the force field
 *
 * @return std::shared_ptr<ForceField>
 */
std::shared_ptr<ForceField> ForceField::clone() const
{
    return std::make_shared<ForceField>(*this);
}

/**
 * @brief find bond type by id
 *
 * @param id
 * @return const BondType&
 *
 * @throws TopologyException if bond type with id not found
 */
const BondType &ForceField::findBondTypeById(const size_t id) const
{
    auto isBondId = [id](const BondType &bondType)
    { return bondType.getId() == id; };

    const auto bondType = std::ranges::find_if(_bondTypes, isBondId);

    if (bondType != _bondTypes.end())
        return *bondType;
    else
        throw TopologyException(
            std::format("Bond type with id {} not found.", id)
        );
}

/**
 * @brief find angle type by id
 *
 * @param id
 * @return const AngleType&
 *
 * @throws TopologyException if angle type with id not found
 */
const AngleType &ForceField::findAngleTypeById(const size_t id) const
{
    auto isAngleId = [id](const AngleType &angleType)
    { return angleType.getId() == id; };

    const auto angleType = std::ranges::find_if(_angleTypes, isAngleId);

    if (angleType != _angleTypes.end())
        return *angleType;
    else
        throw TopologyException(
            std::format("Angle type with id {} not found.", id)
        );
}

/**
 * @brief find dihedral type by id
 *
 * @param id
 * @return const DihedralType&
 *
 * @throws TopologyException if dihedral type with id not found
 */
const DihedralType &ForceField::findDihedralTypeById(const size_t id) const
{
    auto isDihedralId = [id](const DihedralType &dihedralType)
    { return dihedralType.getId() == id; };

    auto      &dihedrals    = _dihedralTypes;
    const auto dihedralType = std::ranges::find_if(dihedrals, isDihedralId);

    if (dihedralType != dihedrals.end())
        return *dihedralType;
    else
        throw TopologyException(
            std::format("Dihedral type with id {} not found.", id)
        );
}

/**
 * @brief find improper dihedral type by id
 *
 * @param id
 * @return const DihedralType&
 *
 * @throws TopologyException if improper dihedral type with id
 * not found
 */
const DihedralType &ForceField::findImproperTypeById(const size_t id) const
{
    auto isImproperId = [id](const DihedralType &dihedralType)
    { return dihedralType.getId() == id; };

    auto      &impropers    = _improperDihedralTypes;
    const auto dihedralType = std::ranges::find_if(impropers, isImproperId);

    if (dihedralType != impropers.end())
        return *dihedralType;
    else
        throw TopologyException(
            std::format("Improper dihedral type with id {} not found.", id)
        );
}

/**
 * @brief find j-coupling type by id
 *
 * @param id
 * @return const JCouplingType&
 *
 * @throws TopologyException if j-coupling type with id not
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
        throw TopologyException(
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
    const SimulationBox &box,
    PhysicalData        &physicalData
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
    const SimulationBox &box,
    PhysicalData        &physicalData
)
{
    auto calculateBondInteraction = [&box, &physicalData, this](auto &bond)
    {
        bond.calculateEnergyAndForces(
            box,
            physicalData,
            *_coulombPotential,
            *_nonCoulombPot
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
    const SimulationBox &box,
    PhysicalData        &physicalData
)
{
    auto calculateAngleInteraction = [&box, &physicalData, this](auto &angle)
    {
        angle.calculateEnergyAndForces(
            box,
            physicalData,
            *_coulombPotential,
            *_nonCoulombPot
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
    const SimulationBox &box,
    PhysicalData        &physicalData
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
            *_nonCoulombPot
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
    const SimulationBox &box,
    PhysicalData        &physicalData
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
            *_nonCoulombPot
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
    const SimulationBox &box,
    PhysicalData        &physicalData
)
{
    if (!_jCouplings.empty())
        throw UserInputException(
            "JCoupling interactions are not implemented yet."
        );

    // auto calculateJCouplingInteraction =
    //     [&box, &physicalData, this](auto &jCoupling)
    // {
    //     jCoupling.calculateEnergyAndForces(
    //         box,
    //         physicalData,
    //         *_coulombPotential,
    //         *_nonCoulombPot
    //     );
    // };

    // std::ranges::for_each(_jCouplings, calculateJCouplingInteraction);
}

/*****************************
 *                           *
 * standard activate methods *
 *                           *
 *****************************/

/**
 * @brief activate non-coulombic interactions
 */
void ForceField::activateNonCoulombic() { _isNonCoulombicActivated = true; }

/**
 * @brief deactivate non-coulombic interactions
 */
void ForceField::deactivateNonCoulombic() { _isNonCoulombicActivated = false; }

/**
 * @brief check if non-coulombic interactions are activated
 *
 * @return bool
 */
bool ForceField::isNonCoulombicActivated() const
{
    return _isNonCoulombicActivated;
}

/***********************************
 *                                 *
 * standard add ForceField Objects *
 *                                 *
 ***********************************/

/**
 * @brief add bond to force field
 *
 * @param bond
 */
void ForceField::addBond(const BondForceField &bond) { _bonds.push_back(bond); }

/**
 * @brief add angle to force field
 *
 * @param angle
 */
void ForceField::addAngle(const AngleForceField &angle)
{
    _angles.push_back(angle);
}

/**
 * @brief add dihedral to force field
 *
 * @param dihedral
 */
void ForceField::addDihedral(const DihedralForceField &dihedral)
{
    _dihedrals.push_back(dihedral);
}

/**
 * @brief add improper dihedral to force field
 *
 * @param improperDihedral
 */
void ForceField::addImproperDihedral(const DihedralForceField &improperDihedral)
{
    _improperDihedrals.push_back(improperDihedral);
}

/**
 * @brief add j-coupling to force field
 *
 * @param jCoupling
 */
void ForceField::addJCoupling(const JCouplingForceField &jCoupling)
{
    _jCouplings.push_back(jCoupling);
}

/***************************************
 *                                     *
 * standard add ForceFieldType objects *
 *                                     *
 ***************************************/

/**
 * @brief add bond type
 *
 * @param bondType
 */
void ForceField::addBondType(const BondType &bondType)
{
    _bondTypes.push_back(bondType);
}

/**
 * @brief add angle type
 *
 * @param angleType
 */
void ForceField::addAngleType(const AngleType &angleType)
{
    _angleTypes.push_back(angleType);
}

/**
 * @brief add dihedral type
 *
 * @param dihedralType
 */
void ForceField::addDihedralType(const DihedralType &dihedralType)
{
    _dihedralTypes.push_back(dihedralType);
}

/**
 * @brief add improper dihedral type
 *
 * @param improperType
 */
void ForceField::addImproperDihedralType(const DihedralType &improperType)
{
    _improperDihedralTypes.push_back(improperType);
}

/**
 * @brief add j-coupling type
 *
 * @param jCouplingType
 */
void ForceField::addJCouplingType(const JCouplingType &jCouplingType)
{
    _jCouplingTypes.push_back(jCouplingType);
}

/**************************
 *                        *
 * standard clear methods *
 *                        *
 **************************/

/**
 * @brief clear bond types
 */
void ForceField::clearBondTypes() { _bondTypes.clear(); }

/**
 * @brief clear angle types
 */
void ForceField::clearAngleTypes() { _angleTypes.clear(); }

/**
 * @brief clear dihedral types
 */
void ForceField::clearDihedralTypes() { _dihedralTypes.clear(); }

/**
 * @brief clear improper dihedral types
 */
void ForceField::clearImproperDihedralTypes()
{
    _improperDihedralTypes.clear();
}

/**
 * @brief clear j-coupling types
 */
void ForceField::clearJCouplingTypes() { _jCouplingTypes.clear(); }

/********************
 *                  *
 * standard setters *
 *                  *
 ********************/

/**
 * @brief set non-coulomb potential
 *
 * @param pot
 */
void ForceField::setNonCoulombPotential(
    const std::shared_ptr<NonCoulombPotential> &pot
)
{
    _nonCoulombPot = pot;
}

/**
 * @brief set coulomb potential
 *
 * @param pot
 */
void ForceField::setCoulombPotential(
    const std::shared_ptr<CoulombPotential> &pot
)
{
    _coulombPotential = pot;
}

/********************
 *                  *
 * standard getters *
 *                  *
 ********************/

/**
 * @brief get bonds
 *
 * @return std::vector<BondForceField>&
 */
std::vector<BondForceField> &ForceField::getBonds() { return _bonds; }

/**
 * @brief get angles
 *
 * @return std::vector<AngleForceField>&
 */
std::vector<AngleForceField> &ForceField::getAngles() { return _angles; }

/**
 * @brief get dihedrals
 *
 * @return std::vector<DihedralForceField>&
 */
std::vector<DihedralForceField> &ForceField::getDihedrals()
{
    return _dihedrals;
}

/**
 * @brief get improper dihedrals
 *
 * @return std::vector<DihedralForceField>&
 */
std::vector<DihedralForceField> &ForceField::getImproperDihedrals()
{
    return _improperDihedrals;
}

/**
 * @brief get j-couplings
 *
 * @return std::vector<JCouplingForceField>&
 */
std::vector<JCouplingForceField> &ForceField::getJCouplings()
{
    return _jCouplings;
}

/**
 * @brief get bond types
 *
 * @return const std::vector<BondType>&
 */
const std::vector<BondType> &ForceField::getBondTypes() const
{
    return _bondTypes;
}

/**
 * @brief get angle types
 *
 * @return const std::vector<AngleType>&
 */
const std::vector<AngleType> &ForceField::getAngleTypes() const
{
    return _angleTypes;
}

/**
 * @brief get dihedral types
 *
 * @return const std::vector<DihedralType>&
 */
const std::vector<DihedralType> &ForceField::getDihedralTypes() const
{
    return _dihedralTypes;
}

/**
 * @brief get improper dihedral types
 *
 * @return const std::vector<DihedralType>&
 */
const std::vector<DihedralType> &ForceField::getImproperTypes() const
{
    return _improperDihedralTypes;
}

/**
 * @brief get j-coupling types
 *
 * @return const std::vector<JCouplingType>&
 */
const std::vector<JCouplingType> &ForceField::getJCouplTypes() const
{
    return _jCouplingTypes;
}