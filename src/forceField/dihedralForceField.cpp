#include "dihedralForceField.hpp"

#include "coulombPotential.hpp"      // for CoulombPotential
#include "molecule.hpp"              // for Molecule
#include "nonCoulombPair.hpp"        // for NonCoulombPair
#include "nonCoulombPotential.hpp"   // for NonCoulombPotential
#include "physicalData.hpp"          // for PhysicalData
#include "potentialSettings.hpp"
#include "simulationBox.hpp"   // for SimulationBox
#include "vector3d.hpp"        // for cross, dot, norm, normSquared

#include <cmath>    // for acos, cos, sin, sqrt
#include <memory>   // for shared_ptr, __shared_ptr_access

using namespace forceField;

/**
 * @brief calculate energy and forces for a single dihedral
 *
 * @details if dihedral is a linker dihedral, correct coulomb and non-coulomb energy and forces (only for non improper dihedrals)
 *
 * @param box
 * @param physicalData
 */
void DihedralForceField::calculateEnergyAndForces(const simulationBox::SimulationBox &box,
                                                  physicalData::PhysicalData         &physicalData,
                                                  const bool                          isImproperDihedral,
                                                  const potential::CoulombPotential  &coulombPotential,
                                                  potential::NonCoulombPotential     &nonCoulombPotential)
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

    const auto distance23 = norm(dPosition23);

    auto phi = angle(crossPosition123, crossPosition432);
    phi      = dot(dPosition12, crossPosition432) > 0.0 ? -phi : phi;

    if (isImproperDihedral)
        physicalData.addImproperEnergy(_forceConstant * (1.0 + ::cos(_periodicity * phi + _phaseShift)));
    else
        physicalData.addDihedralEnergy(_forceConstant * (1.0 + ::cos(_periodicity * phi + _phaseShift)));

    auto       forceMagnitude = distance23 / distance123Squared;
    const auto forceVector12  = forceMagnitude * crossPosition123;

    forceMagnitude           = distance23 / distance432Squared;
    const auto forceVector43 = forceMagnitude * crossPosition432;

    forceMagnitude            = dot(dPosition12, dPosition23) / (distance123Squared * distance23);
    const auto forceVector123 = forceMagnitude * crossPosition123;

    forceMagnitude            = dot(dPosition43, dPosition23) / (distance432Squared * distance23);
    const auto forceVector432 = forceMagnitude * crossPosition432;

    forceMagnitude = _forceConstant * _periodicity * ::sin(_periodicity * phi + _phaseShift);

    _molecules[0]->addAtomForce(_atomIndices[0], -forceMagnitude * forceVector12);
    _molecules[1]->addAtomForce(_atomIndices[1], +forceMagnitude * (forceVector12 + forceVector123 - forceVector432));
    _molecules[2]->addAtomForce(_atomIndices[2], +forceMagnitude * (-forceVector43 - forceVector123 + forceVector432));
    _molecules[3]->addAtomForce(_atomIndices[3], +forceMagnitude * forceVector43);

    if (_isLinker)
    {
        auto dPosition14 = position1 - position4;
        box.applyPBC(dPosition14);

        const auto distance14 = norm(dPosition14);

        if (distance14 < potential::CoulombPotential::getCoulombRadiusCutOff())
        {
            const auto chargeProduct =
                _molecules[0]->getPartialCharge(_atomIndices[0]) * _molecules[3]->getPartialCharge(_atomIndices[3]);

            const auto [coulombEnergy, coulombForce] = coulombPotential.calculate(distance14, chargeProduct);

            forceMagnitude = -coulombForce * (1.0 - settings::PotentialSettings::getScale14Coulomb());
            physicalData.addCoulombEnergy(-coulombEnergy * (1.0 - settings::PotentialSettings::getScale14Coulomb()));

            const auto molType1  = _molecules[0]->getMoltype();
            const auto molType2  = _molecules[3]->getMoltype();
            const auto atomType1 = _molecules[0]->getAtomType(_atomIndices[0]);
            const auto atomType2 = _molecules[3]->getAtomType(_atomIndices[3]);
            const auto vdwType1  = _molecules[0]->getInternalGlobalVDWType(_atomIndices[0]);
            const auto vdwType2  = _molecules[3]->getInternalGlobalVDWType(_atomIndices[3]);

            const auto combinedIndices = {molType1, molType2, atomType1, atomType2, vdwType1, vdwType2};

            if (const auto nonCoulombPair = nonCoulombPotential.getNonCoulombPair(combinedIndices);
                distance14 < nonCoulombPair->getRadialCutOff())
            {
                const auto [nonCoulombEnergy, nonCoulombForce] = nonCoulombPair->calculateEnergyAndForce(distance14);

                forceMagnitude -= nonCoulombForce * (1.0 - settings::PotentialSettings::getScale14VanDerWaals());
                physicalData.addNonCoulombEnergy(-nonCoulombEnergy *
                                                 (1.0 - settings::PotentialSettings::getScale14VanDerWaals()));
            }

            forceMagnitude /= distance14;

            const auto forcexyz = forceMagnitude * dPosition14;

            physicalData.addVirial(forcexyz * dPosition14);

            _molecules[0]->addAtomForce(_atomIndices[0], forcexyz);
            _molecules[3]->addAtomForce(_atomIndices[3], -forcexyz);
        }
    }
}
