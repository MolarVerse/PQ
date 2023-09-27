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

#include "intraNonBonded.hpp"

#include "exceptions.hpp"
#include "simulationBox.hpp"

#include <algorithm>    // for for_each
#include <format>       // for format
#include <functional>   // for identity
#include <ranges>       // for std::ranges::find_if
#include <string>       // for string

using namespace intraNonBonded;

/**
 * @brief find a intraNonBondedContainer by molType and return a pointer to it
 *
 * @param molType
 * @return IntraNonBondedContainer*
 */
IntraNonBondedContainer *IntraNonBonded::findIntraNonBondedContainerByMolType(const size_t molType)
{
    auto findByMolType = [molType](const auto &intraNonBondedType) { return intraNonBondedType.getMolType() == molType; };

    if (const auto it = std::ranges::find_if(_intraNonBondedContainers, findByMolType); it != _intraNonBondedContainers.end())
        return std::to_address(it);
    else
        throw customException::IntraNonBondedException(
            std::format("IntraNonBondedContainer with molType {} not found!", molType));
}

/**
 * @brief fill the _intraNonBondedMaps vector with IntraNonBondedMap objects
 *
 * @param box
 */
void IntraNonBonded::fillIntraNonBondedMaps(simulationBox::SimulationBox &box)
{
    auto fillSingleMap = [this](auto &molecule)
    {
        auto *intraNonBondedContainer = findIntraNonBondedContainerByMolType(molecule.getMoltype());
        _intraNonBondedMaps.push_back(IntraNonBondedMap(&molecule, intraNonBondedContainer));
    };

    std::ranges::for_each(box.getMolecules(), fillSingleMap);
}

/**
 * @brief calculate the intra non bonded interactions for each intraNonBondedMap
 *
 * @param box
 * @param physicalData
 */
void IntraNonBonded::calculate(const simulationBox::SimulationBox &box, physicalData::PhysicalData &physicalData)
{
    auto calculateSingleContribution = [this, &box, &physicalData](auto &intraNonBondedMap)
    { intraNonBondedMap.calculate(_coulombPotential.get(), _nonCoulombPotential.get(), box, physicalData); };

    std::ranges::for_each(_intraNonBondedMaps, calculateSingleContribution);
}