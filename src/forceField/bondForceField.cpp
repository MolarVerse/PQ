#include "bondForceField.hpp"

#include "coulombPotential.hpp"
#include "molecule.hpp"
#include "nonCoulombPair.hpp"
#include "nonCoulombPotential.hpp"
#include "physicalData.hpp"
#include "simulationBox.hpp"
#include "vector3d.hpp"

#include <memory>
#include <vector>

using namespace forceField;
using namespace simulationBox;
using namespace physicalData;
using namespace potential;

/**
 * @brief calculate energy and forces for a single bond
 *
 * @param box
 * @param physicalData
 */
void BondForceField::calculateEnergyAndForces(const SimulationBox    &box,
                                              PhysicalData           &physicalData,
                                              const CoulombPotential &coulombPotential,
                                              NonCoulombPotential    &nonCoulombPotential)
{
    const auto position1 = _molecules[0]->getAtomPosition(_atomIndices[0]);
    const auto position2 = _molecules[1]->getAtomPosition(_atomIndices[1]);
    auto       dPosition = position1 - position2;

    box.applyPBC(dPosition);

    const auto distance      = norm(dPosition);
    const auto deltaDistance = distance - _equilibriumBondLength;

    auto forceMagnitude = -_forceConstant * deltaDistance;

    physicalData.addBondEnergy(-forceMagnitude * deltaDistance / 2.0);

    if (_isLinker && distance < CoulombPotential::getCoulombRadiusCutOff())
    {
        const auto chargeProduct =
            _molecules[0]->getPartialCharge(_atomIndices[0]) * _molecules[1]->getPartialCharge(_atomIndices[1]);

        const auto [coulombEnergy, coulombForce] = coulombPotential.calculate(distance, chargeProduct);

        forceMagnitude -= coulombForce;
        physicalData.addCoulombEnergy(-coulombEnergy);

        const auto molType1  = _molecules[0]->getMoltype();
        const auto molType2  = _molecules[1]->getMoltype();
        const auto atomType1 = _molecules[0]->getAtomType(_atomIndices[0]);
        const auto atomType2 = _molecules[1]->getAtomType(_atomIndices[1]);
        const auto vdwType1  = _molecules[0]->getInternalGlobalVDWType(_atomIndices[0]);
        const auto vdwType2  = _molecules[1]->getInternalGlobalVDWType(_atomIndices[1]);

        const auto combinedIndices = {molType1, molType2, atomType1, atomType2, vdwType1, vdwType2};

        if (const auto nonCoulombPair = nonCoulombPotential.getNonCoulombPair(combinedIndices);
            distance < nonCoulombPair->getRadialCutOff())
        {
            const auto [nonCoulombEnergy, nonCoulombForce] = nonCoulombPair->calculateEnergyAndForce(distance);

            forceMagnitude -= nonCoulombForce;
            physicalData.addNonCoulombEnergy(-nonCoulombEnergy);
        }
    }

    forceMagnitude /= distance;

    const auto force = forceMagnitude * dPosition;

    _molecules[0]->addAtomForce(_atomIndices[0], force);
    _molecules[1]->addAtomForce(_atomIndices[1], -force);

    physicalData.addVirial(force * dPosition);
}