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

void IntraNonBonded::calculate(const simulationBox::SimulationBox &box, physicalData::PhysicalData &physicalData)
{
    auto calculateSingleContribution = [this, &box, &physicalData](auto &intraNonBondedMap)
    { intraNonBondedMap.calculate(_coulombPotential.get(), _nonCoulombPotential.get(), box, physicalData); };

    std::ranges::for_each(_intraNonBondedMaps, calculateSingleContribution);
}