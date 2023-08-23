#include "dihedralForceField.hpp"

using namespace forceField;
using namespace simulationBox;
using namespace physicalData;

/**
 * @brief calculate energy and forces for a single dihedral
 *
 * @param box
 * @param physicalData
 */
void DihedralForceField::calculateEnergyAndForces(const SimulationBox &box, PhysicalData &physicalData, bool isImproper)
{
    const auto position2 = _molecules[1]->getAtomPosition(_atomIndices[1]);   // central position of dihedral
    const auto position3 = _molecules[2]->getAtomPosition(_atomIndices[2]);   // central position of dihedral

    const auto position1 = _molecules[0]->getAtomPosition(_atomIndices[0]);
    const auto position4 = _molecules[3]->getAtomPosition(_atomIndices[3]);

    auto dPosition12 = position1 - position2;
    auto dPosition23 = position2 - position3;
    auto dPosition43 = position4 - position3;

    box.applyPBC(dPosition12);
    box.applyPBC(dPosition23);
    box.applyPBC(dPosition43);

    auto crossPosition123 = cross(dPosition12, dPosition23);
    auto crossPosition432 = cross(dPosition43, dPosition23);

    const auto distance123Squared = normSquared(crossPosition123);
    const auto distance432Squared = normSquared(crossPosition432);

    const auto distance23  = norm(dPosition23);
    const auto distance123 = ::sqrt(distance123Squared);
    const auto distance432 = ::sqrt(distance432Squared);

    auto       forceMagnitude = distance23 / distance123Squared;
    const auto forceVector12  = forceMagnitude * crossPosition123;
    forceMagnitude            = distance23 / distance432Squared;
    const auto forceVector43  = forceMagnitude * crossPosition432;

    forceMagnitude            = dot(dPosition12, dPosition23) / (distance123Squared * distance23);
    const auto forceVector123 = forceMagnitude * crossPosition123;
    forceMagnitude            = dot(dPosition43, dPosition23) / (distance432Squared * distance23);
    const auto forceVector432 = forceMagnitude * crossPosition432;

    auto cosine = dot(crossPosition123, crossPosition432) / (distance123 * distance432);

    cosine = cosine > 1.0 ? 1.0 : cosine;
    cosine = cosine < -1.0 ? -1.0 : cosine;

    auto angle = ::acos(cosine);

    angle = dot(dPosition12, crossPosition432) > 0.0 ? -angle : angle;

    if (isImproper)
        physicalData.addImproperEnergy(_forceConstant * (1.0 + ::cos(_periodicity * angle + _phaseShift)));
    else
        physicalData.addDihedralEnergy(_forceConstant * (1.0 + ::cos(_periodicity * angle + _phaseShift)));

    forceMagnitude = _forceConstant * _periodicity * ::sin(_periodicity * angle + _phaseShift);

    _molecules[0]->addAtomForce(_atomIndices[0], -forceMagnitude * forceVector12);
    _molecules[1]->addAtomForce(_atomIndices[1], +forceMagnitude * (forceVector12 + forceVector123 - forceVector432));
    _molecules[2]->addAtomForce(_atomIndices[2], +forceMagnitude * (-forceVector43 - forceVector123 + forceVector432));
    _molecules[3]->addAtomForce(_atomIndices[3], +forceMagnitude * forceVector43);
}
