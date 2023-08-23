#include "intraNonBondedMap.hpp"

using namespace intraNonBonded;

void IntraNonBondedMap::calculate(const potential::CoulombPotential  *coulombPotential,
                                  potential::NonCoulombPotential     *nonCoulombPotential,
                                  const simulationBox::SimulationBox &box,
                                  physicalData::PhysicalData         &physicalData)
{
    auto coulombEnergy    = 0.0;
    auto nonCoulombEnergy = 0.0;

    for (const auto &atomIndices : _intraNonBondedType->getAtomIndices())
    {
        const auto &pos1 = _molecule->getAtomPosition(size_t(atomIndices[0]));

        for (auto iter = atomIndices.begin() + 1; iter != atomIndices.end(); ++iter)
        {
            const auto atomIndex = size_t(::abs(*iter));
            const bool scale     = *iter < 0;

            const auto &pos2 = _molecule->getAtomPosition(atomIndex);

            auto dPos = pos1 - pos2;
            box.applyPBC(dPos);

            const auto distance = norm(dPos);

            if (distance < potential::CoulombPotential::getCoulombRadiusCutOff())   // TODO: check if needed
            {

                const auto chargeProduct =
                    _molecule->getPartialCharge(size_t(atomIndices[0])) * _molecule->getPartialCharge(atomIndex);

                auto [energy, force] = coulombPotential->calculate(distance, chargeProduct);

                if (scale)
                {
                    energy *= _scale14Coulomb;
                    force  *= _scale14Coulomb;
                }
                coulombEnergy += energy;

                const size_t atomType = _molecule->getAtomType(atomIndex);

                const auto globalVdwType = _molecule->getInternalGlobalVDWType(atomIndex);

                const auto moltype = _molecule->getMoltype();

                const auto combinedIndices = {moltype, moltype, atomType, atomType, globalVdwType, globalVdwType};

                const auto nonCoulombicPair = nonCoulombPotential->getNonCoulombPair(combinedIndices);

                if (distance < nonCoulombicPair->getRadialCutOff())
                {
                    auto [energy, nonCoulombForce] = nonCoulombicPair->calculateEnergyAndForce(distance);

                    if (scale)
                    {
                        energy          *= _scale14VanDerWaals;
                        nonCoulombForce *= _scale14VanDerWaals;
                    }

                    nonCoulombEnergy += energy;
                    force            += nonCoulombForce;
                }

                force /= distance;

                const auto forcexyz = force * dPos;

                // const auto shiftForcexyz = forcexyz * txyz;

                _molecule->addAtomForce(size_t(atomIndices[0]), forcexyz);
                _molecule->addAtomForce(atomIndex, -forcexyz);

                // molecule1.addAtomShiftForce(atom1, shiftForcexyz);
            }
        }
    }

    physicalData.addIntraCoulombEnergy(coulombEnergy);
    physicalData.addIntraNonCoulombEnergy(nonCoulombEnergy);
}