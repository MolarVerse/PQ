#include "intraNonBonded.hpp"

#include "exceptions.hpp"

#include <ranges>

using namespace std;
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
    if (const auto it = ranges::find_if(_intraNonBondedContainers, findByMolType); it != _intraNonBondedContainers.end())
        return std::to_address(it);
    else
        throw customException::IntraNonBondedException(format("IntraNonBondedContainer with molType {} not found!", molType));
}

/**
 * @brief fill the _intraNonBondedMaps vector with IntraNonBondedMap objects
 *
 * @param box
 */
void IntraNonBonded::fillIntraNonBondedMaps(simulationBox::SimulationBox &box)
{
    auto fillSingleMap = [this](auto molecule)
    {
        auto *intraNonBondedContainer = findIntraNonBondedContainerByMolType(molecule.getMoltype());
        _intraNonBondedMaps.push_back(IntraNonBondedMap(&molecule, intraNonBondedContainer));
    };

    ranges::for_each(box.getMolecules(), fillSingleMap);
}

/**
 * @brief calculate the energy and forces of the intra non bonded interactions
 *
 * @param box
 * @param physicalData
 */
void IntraNonBonded::calculateEnergyAndForces(simulationBox::SimulationBox &box,
                                              forceField::ForceField       &forceField,
                                              physicalData::PhysicalData   &physicalData)
{
    if (!_isActivated)
        return;

    auto nonCoulombicPairMatrix = forceField.getNonCoulombicPairsMatrix();

    for (auto &intraNonBondedMap : _intraNonBondedMaps)
    {
        const auto atomIndices = intraNonBondedMap.getAtomIndices();
        auto      *molecule    = intraNonBondedMap.getMolecule();

        for (size_t atom_2 = 0; atom_2 < atomIndices.size(); ++atom_2)
        {
            const auto externalGlobalVdwType1 = molecule->getExternalGlobalVDWType(atom_2);
            const auto internalGlobalVdwType1 = box.getExternalToInternalGlobalVDWTypes().at(externalGlobalVdwType1);

            for (const auto atom_2 : atomIndices[atom_2])
            {
                const auto externalGlobalVdwType2 = molecule->getExternalGlobalVDWType(atom_2);
                const auto internalGlobalVdwType2 = box.getExternalToInternalGlobalVDWTypes().at(externalGlobalVdwType2);

                auto nonCoulombicPair = nonCoulombicPairMatrix[internalGlobalVdwType1][internalGlobalVdwType2];
            }
        }
    }
}