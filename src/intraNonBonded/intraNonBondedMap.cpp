#include "intraNonBondedMap.hpp"

#include "coulombPotential.hpp"      // for CoulombPotential
#include "molecule.hpp"              // for Molecule
#include "nonCoulombPair.hpp"        // for NonCoulombPair
#include "nonCoulombPotential.hpp"   // for NonCoulombPotential
#include "physicalData.hpp"          // for PhysicalData
#include "potentialSettings.hpp"     // for PotentialSettings
#include "simulationBox.hpp"         // for SimulationBox
#include "vector3d.hpp"              // for norm, operator*, Vector3D

#include <cstdlib>   // for abs, size_t
#include <memory>    // for __shared_ptr_access, shared_ptr

using namespace intraNonBonded;

/**
 * @brief calculate the intra non bonded interactions for a single intraNonBondedMap (for a single molecule)
 *
 * @param coulombPotential
 * @param nonCoulombPotential
 * @param box
 * @param physicalData
 */
void IntraNonBondedMap::calculate(const potential::CoulombPotential  *coulombPotential,
                                  potential::NonCoulombPotential     *nonCoulombPotential,
                                  const simulationBox::SimulationBox &simulationBox,
                                  physicalData::PhysicalData         &physicalData) const
{
    auto       coulombEnergy    = 0.0;
    auto       nonCoulombEnergy = 0.0;
    const auto box              = simulationBox.getBoxDimensions();

    for (size_t atomIndex1 = 0; atomIndex1 < _intraNonBondedContainer->getAtomIndices().size(); ++atomIndex1)
    {
        const auto atomIndices = _intraNonBondedContainer->getAtomIndices()[atomIndex1];

        for (auto iter = atomIndices.begin(); iter != atomIndices.end(); ++iter)
        {
            const auto [coulombEnergyTemp, nonCoulombEnergyTemp] =
                calculateSingleInteraction(atomIndex1, *iter, box, physicalData, coulombPotential, nonCoulombPotential);

            coulombEnergy    += coulombEnergyTemp;
            nonCoulombEnergy += nonCoulombEnergyTemp;
        }
    }

    physicalData.addIntraCoulombEnergy(coulombEnergy);
    physicalData.addIntraNonCoulombEnergy(nonCoulombEnergy);
}

/**
 * @brief calculate the intra non bonded interactions for a single atomic pair within a single molecule
 *
 * @param atomIndex1
 * @param atomIndex2AsInt
 * @param box
 * @param coulombPotential
 * @param nonCoulombPotential
 * @return std::pair<double, double> - the coulomb and non-coulomb energy for the interaction
 */
std::pair<double, double> IntraNonBondedMap::calculateSingleInteraction(const size_t                       atomIndex1,
                                                                        const int                          atomIndex2AsInt,
                                                                        const linearAlgebra::Vec3D        &box,
                                                                        physicalData::PhysicalData        &physicalData,
                                                                        const potential::CoulombPotential *coulombPotential,
                                                                        potential::NonCoulombPotential *nonCoulombPotential) const
{
    auto coulombEnergy    = 0.0;
    auto nonCoulombEnergy = 0.0;

    const auto atomIndex2 = size_t(::abs(atomIndex2AsInt));
    const bool scale      = atomIndex2AsInt < 0;

    const auto &pos1 = _molecule->getAtomPosition(atomIndex1);
    const auto &pos2 = _molecule->getAtomPosition(atomIndex2);

    auto       dPos = pos1 - pos2;
    const auto txyz = -box * round(dPos / box);

    dPos += txyz;

    if (const auto distance = norm(dPos); distance < potential::CoulombPotential::getCoulombRadiusCutOff())
    {
        const auto chargeProduct = _molecule->getPartialCharge(size_t(atomIndex1)) * _molecule->getPartialCharge(atomIndex2);

        auto [energy, force] = coulombPotential->calculate(distance, chargeProduct);

        if (scale)
        {
            energy *= settings::PotentialSettings::getScale14Coulomb();
            force  *= settings::PotentialSettings::getScale14Coulomb();
        }
        coulombEnergy = energy;

        const size_t atomType1 = _molecule->getAtomType(atomIndex1);
        const size_t atomType2 = _molecule->getAtomType(atomIndex2);

        const auto globalVdwType1 = _molecule->getInternalGlobalVDWType(atomIndex1);
        const auto globalVdwType2 = _molecule->getInternalGlobalVDWType(atomIndex2);

        const auto moltype = _molecule->getMoltype();

        const auto combinedIndices = {moltype, moltype, atomType1, atomType2, globalVdwType1, globalVdwType2};

        if (const auto nonCoulombicPair = nonCoulombPotential->getNonCoulombPair(combinedIndices);
            distance < nonCoulombicPair->getRadialCutOff())
        {

            auto [nonCoulombEnergyLocal, nonCoulombForce] = nonCoulombicPair->calculateEnergyAndForce(distance);

            if (scale)
            {
                nonCoulombEnergyLocal *= settings::PotentialSettings::getScale14VanDerWaals();
                nonCoulombForce       *= settings::PotentialSettings::getScale14VanDerWaals();
            }

            nonCoulombEnergy  = nonCoulombEnergyLocal;
            force            += nonCoulombForce;
        }

        force /= distance;

        const auto forcexyz = force * dPos;

        const auto shiftForcexyz = forcexyz * txyz;

        physicalData.addVirial(forcexyz * txyz);

        _molecule->addAtomForce(atomIndex1, forcexyz);
        _molecule->addAtomForce(atomIndex2, -forcexyz);

        _molecule->addAtomShiftForce(atomIndex1, shiftForcexyz);
    }

    return {coulombEnergy, nonCoulombEnergy};
}