#include "angleForceField.hpp"

using namespace forceField;
using namespace simulationBox;
using namespace physicalData;
using namespace potential;

/**
 * @brief calculate energy and forces for a single angle
 *
 * @param box
 * @param physicalData
 */
void AngleForceField::calculateEnergyAndForces(const SimulationBox    &box,
                                               PhysicalData           &physicalData,
                                               const CoulombPotential &coulombPotential,
                                               NonCoulombPotential    &nonCoulombPotential)
{

    const auto position1 = _molecules[0]->getAtomPosition(_atomIndices[0]);   // central position of angle
    const auto position2 = _molecules[1]->getAtomPosition(_atomIndices[1]);
    const auto position3 = _molecules[2]->getAtomPosition(_atomIndices[2]);

    auto dPosition12 = position1 - position2;
    auto dPosition13 = position1 - position3;

    box.applyPBC(dPosition12);
    box.applyPBC(dPosition13);

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

    physicalData.addAngleEnergy(-forceMagnitude * deltaAngle / 2.0);

    const auto normalDistance = distance12 * distance13 * ::sin(angle);
    const auto normalPosition = cross(dPosition13, dPosition12) / normalDistance;

    auto force    = forceMagnitude / distance12Squared;
    auto forcexyz = force * cross(dPosition12, normalPosition);

    _molecules[0]->addAtomForce(_atomIndices[0], -forcexyz);
    _molecules[1]->addAtomForce(_atomIndices[1], forcexyz);

    force    = forceMagnitude / distance13Squared;
    forcexyz = force * cross(normalPosition, dPosition13);

    _molecules[0]->addAtomForce(_atomIndices[0], -forcexyz);
    _molecules[2]->addAtomForce(_atomIndices[2], forcexyz);

    if (_isLinker)
    {
        auto dPosition23 = position2 - position3;
        box.applyPBC(dPosition23);

        const auto distance23 = norm(dPosition23);

        // if (distance23 < CoulombPotential::getCoulombRadiusCutOff())
        {
            const auto chargeProduct =
                _molecules[1]->getPartialCharge(_atomIndices[1]) * _molecules[2]->getPartialCharge(_atomIndices[2]);

            const auto [coulombEnergy, coulombForce] = coulombPotential.calculate(distance23, chargeProduct);

            forceMagnitude = -coulombForce;
            physicalData.addCoulombEnergy(-coulombEnergy);

            const auto molType1  = _molecules[1]->getMoltype();
            const auto molType2  = _molecules[2]->getMoltype();
            const auto atomType1 = _molecules[1]->getAtomType(_atomIndices[1]);
            const auto atomType2 = _molecules[2]->getAtomType(_atomIndices[2]);
            const auto vdwType1  = _molecules[1]->getInternalGlobalVDWType(_atomIndices[1]);
            const auto vdwType2  = _molecules[2]->getInternalGlobalVDWType(_atomIndices[2]);

            const auto combinedIndices = {molType1, molType2, atomType1, atomType2, vdwType1, vdwType2};

            const auto nonCoulombPair = nonCoulombPotential.getNonCoulombPair(combinedIndices);

            if (distance23 < 0)   // nonCoulombPair->getRadialCutOff())
            {
                const auto [nonCoulombEnergy, nonCoulombForce] = nonCoulombPair->calculateEnergyAndForce(distance23);

                forceMagnitude -= nonCoulombForce;
                physicalData.addNonCoulombEnergy(-nonCoulombEnergy);
            }

            forceMagnitude /= distance23;

            forcexyz = forceMagnitude * dPosition23;

            physicalData.addVirial(forcexyz * dPosition23);

            _molecules[1]->addAtomForce(_atomIndices[1], forcexyz);
            _molecules[2]->addAtomForce(_atomIndices[2], -forcexyz);
        }
    }
}