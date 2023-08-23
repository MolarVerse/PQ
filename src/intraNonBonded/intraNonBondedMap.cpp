#include "intraNonBondedMap.hpp"

using namespace intraNonBonded;

void IntraNonBondedMap::calculate(const potential::CoulombPotential  *coulombPotential,
                                  potential::NonCoulombPotential     *nonCoulombPotential,
                                  const simulationBox::SimulationBox &box,
                                  physicalData::PhysicalData         &physicalData)
{
    auto coulombEnergy    = 0.0;
    auto nonCoulombEnergy = 0.0;

    for (size_t atomIndex1 = 0; atomIndex1 < _intraNonBondedType->getAtomIndices().size(); ++atomIndex1)
    {
        const auto  atomIndices = _intraNonBondedType->getAtomIndices()[atomIndex1];
        const auto &pos1        = _molecule->getAtomPosition(atomIndex1);

        for (auto iter = atomIndices.begin(); iter != atomIndices.end(); ++iter)
        {
            const auto atomIndex2 = size_t(::abs(*iter));
            const bool scale      = *iter < 0;

            const auto &pos2 = _molecule->getAtomPosition(atomIndex2);

            auto dPos = pos1 - pos2;
            box.applyPBC(dPos);

            const auto distance = norm(dPos);

            if (distance < potential::CoulombPotential::getCoulombRadiusCutOff())   // TODO: check if needed
            {

                const auto chargeProduct =
                    _molecule->getPartialCharge(size_t(atomIndex1)) * _molecule->getPartialCharge(atomIndex2);

                auto [energy, force] = coulombPotential->calculate(distance, chargeProduct);

                if (scale)
                {
                    energy *= _scale14Coulomb;
                    force  *= _scale14Coulomb;
                }
                coulombEnergy += energy;

                const size_t atomType = _molecule->getAtomType(atomIndex2);

                const auto globalVdwType = _molecule->getInternalGlobalVDWType(atomIndex2);

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

                _molecule->addAtomForce(size_t(atomIndex1), forcexyz);
                _molecule->addAtomForce(atomIndex2, -forcexyz);

                // molecule1.addAtomShiftForce(atom1, shiftForcexyz);
            }
        }
    }

    physicalData.addIntraCoulombEnergy(coulombEnergy);
    physicalData.addIntraNonCoulombEnergy(nonCoulombEnergy);
}