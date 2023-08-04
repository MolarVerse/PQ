#include "forceField.hpp"

#include "exceptions.hpp"

#include <algorithm>

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
const BondType &ForceField::findBondTypeById(size_t id) const
{
    auto isBondId = [id](const BondType &bondType) { return bondType.getId() == id; };

    if (auto bondType = ranges::find_if(_bondTypes, isBondId); bondType != _bondTypes.end())
        return *bondType;
    else
        throw customException::TopologyException("Bond type with id " + to_string(id) + " not found.");
}

/**
 * @brief find angle type by id
 *
 * @param id
 * @return const AngleType&
 *
 * @throws customException::TopologyException if angle type with id not found
 */
const AngleType &ForceField::findAngleTypeById(size_t id) const
{
    auto isAngleId = [id](const AngleType &angleType) { return angleType.getId() == id; };

    if (auto angleType = ranges::find_if(_angleTypes, isAngleId); angleType != _angleTypes.end())
        return *angleType;
    else
        throw customException::TopologyException("Angle type with id " + to_string(id) + " not found.");
}

/**
 * @brief find dihedral type by id
 *
 * @param id
 * @return const DihedralType&
 *
 * @throws customException::TopologyException if dihedral type with id not found
 */
const DihedralType &ForceField::findDihedralTypeById(size_t id) const
{
    auto isDihedralId = [id](const DihedralType &dihedralType) { return dihedralType.getId() == id; };

    if (auto dihedralType = ranges::find_if(_dihedralTypes, isDihedralId); dihedralType != _dihedralTypes.end())
        return *dihedralType;
    else
        throw customException::TopologyException("Dihedral type with id " + to_string(id) + " not found.");
}

/**
 * @brief find improper dihedral type by id
 *
 * @param id
 * @return const DihedralType&
 *
 * @throws customException::TopologyException if improper dihedral type with id not found
 */
const DihedralType &ForceField::findImproperDihedralTypeById(size_t id) const
{
    auto isImproperDihedralId = [id](const DihedralType &dihedralType) { return dihedralType.getId() == id; };

    if (auto dihedralType = ranges::find_if(_improperDihedralTypes, isImproperDihedralId);
        dihedralType != _improperDihedralTypes.end())
        return *dihedralType;
    else
        throw customException::TopologyException("Improper dihedral type with id " + to_string(id) + " not found.");
}