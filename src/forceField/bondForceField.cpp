#include "bondForceField.hpp"

using namespace forceField;
using namespace simulationBox;
using namespace physicalData;

/**
 * @brief calculate energy and forces for a single bond
 *
 * @param box
 * @param physicalData
 */
void BondForceField::calculateEnergyAndForces(const SimulationBox &box, PhysicalData &physicalData)
{
    const auto position1 = _molecules[0]->getAtomPosition(_atomIndices[0]);
    const auto position2 = _molecules[1]->getAtomPosition(_atomIndices[1]);
    auto       dPosition = position1 - position2;

    box.applyPBC(dPosition);

    const auto distance      = norm(dPosition);
    const auto deltaDistance = distance - _equilibriumBondLength;

    auto forceMagnitude = -_forceConstant * deltaDistance;

    physicalData.addBondEnergy(-forceMagnitude * deltaDistance / 2.0);

    forceMagnitude /= distance;

    const auto force = forceMagnitude * dPosition;

    _molecules[0]->addAtomForce(_atomIndices[0], force);
    _molecules[1]->addAtomForce(_atomIndices[1], -force);

    physicalData.addVirial(force * dPosition);
}