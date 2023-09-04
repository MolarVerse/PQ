#include "bondForceField.hpp"

#include "coulombPotential.hpp"   // for CoulombPotential
#include "forceField.hpp"         // for correctLinker
#include "molecule.hpp"           // for Molecule
#include "physicalData.hpp"       // for PhysicalData
#include "simulationBox.hpp"      // for SimulationBox
#include "vector3d.hpp"           // for Vector3D, norm, operator*, Vec3D

#include <vector>   // for vector

using namespace forceField;

/**
 * @brief calculate energy and forces for a single bond
 *
 * @details if bond is a linker bond, correct coulomb and non-coulomb energy and forces
 *
 * @param box
 * @param physicalData
 */
void BondForceField::calculateEnergyAndForces(const simulationBox::SimulationBox &box,
                                              physicalData::PhysicalData         &physicalData,
                                              const potential::CoulombPotential  &coulombPotential,
                                              potential::NonCoulombPotential     &nonCoulombPotential)
{
    const auto position1 = _molecules[0]->getAtomPosition(_atomIndices[0]);
    const auto position2 = _molecules[1]->getAtomPosition(_atomIndices[1]);
    auto       dPosition = position1 - position2;

    box.applyPBC(dPosition);

    const auto distance      = norm(dPosition);
    const auto deltaDistance = distance - _equilibriumBondLength;

    auto forceMagnitude = -_forceConstant * deltaDistance;

    physicalData.addBondEnergy(-forceMagnitude * deltaDistance / 2.0);

    if (_isLinker && distance < potential::CoulombPotential::getCoulombRadiusCutOff())
    {
        forceMagnitude += correctLinker(coulombPotential,
                                        nonCoulombPotential,
                                        physicalData,
                                        _molecules[0],
                                        _molecules[1],
                                        _atomIndices[0],
                                        _atomIndices[1],
                                        distance,
                                        false);
    }

    forceMagnitude /= distance;

    const auto force = forceMagnitude * dPosition;

    _molecules[0]->addAtomForce(_atomIndices[0], force);
    _molecules[1]->addAtomForce(_atomIndices[1], -force);

    physicalData.addVirial(force * dPosition);
}