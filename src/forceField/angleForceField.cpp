#include "angleForceField.hpp"

using namespace forceField;
using namespace simulationBox;
using namespace physicalData;

/**
 * @brief calculate energy and forces for a single angle
 *
 * @param box
 * @param physicalData
 */
void AngleForceField::calculateEnergyAndForces(const SimulationBox &box, PhysicalData &physicalData)
{
    const auto position1 = _molecules[0]->getAtomPosition(_atomIndices[0]);
    const auto position2 = _molecules[1]->getAtomPosition(_atomIndices[1]);
    const auto position3 = _molecules[2]->getAtomPosition(_atomIndices[2]);

    auto dPosition12 = position1 - position2;
    auto dPosition13 = position1 - position3;
    auto dPosition23 = position2 - position3;

    box.applyPBC(dPosition12);
    box.applyPBC(dPosition13);
    box.applyPBC(dPosition23);

    const auto distance12Squared = normSquared(dPosition12);
    const auto distance13Squared = normSquared(dPosition13);

    const auto distance12 = ::sqrt(distance12Squared);
    const auto distance13 = ::sqrt(distance13Squared);

    auto cosine = dot(dPosition12, dPosition13) / (distance12 * distance13);

    cosine = cosine > 1.0 ? 1.0 : cosine;
    cosine = cosine < -1.0 ? -1.0 : cosine;

    const auto angle = ::acos(cosine);

    const auto deltaAngle = angle - _equilibriumAngle;

    auto forceMagnitude = -_forceConstant * deltaAngle;

    physicalData.setAngleEnergy(-forceMagnitude * deltaAngle / 2.0);

    const auto normalDistance = distance12 * distance13 * ::sin(angle);
    const auto normalPosition = cross(dPosition12, dPosition13) / normalDistance;

    auto       force  = forceMagnitude / distance12Squared;
    const auto force1 = force * cross(dPosition12, normalPosition);

    _molecules[0]->addAtomForce(_atomIndices[0], force1);
    _molecules[1]->addAtomForce(_atomIndices[1], -force1);

    force             = forceMagnitude / distance13Squared;
    const auto force2 = force * cross(normalPosition, dPosition13);

    _molecules[0]->addAtomForce(_atomIndices[0], force2);
    _molecules[2]->addAtomForce(_atomIndices[2], -force2);
}