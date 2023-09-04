#include "forceField.hpp"

#include "coulombPotential.hpp"      // for CoulombPotential
#include "molecule.hpp"              // for Molecule
#include "nonCoulombPair.hpp"        // for NonCoulombPair
#include "nonCoulombPotential.hpp"   // for NonCoulombPotential
#include "physicalData.hpp"          // for PhysicalData
#include "potentialSettings.hpp"     // for PotentialSettings

#include <memory>   // for __shared_ptr_access, shared_ptr

/**
 * @brief correct coulomb and non-coulomb energy and forces for linker connectivity elements
 *
 * @param coulombPotential
 * @param nonCoulombPotential
 * @param physicalData
 * @param molecule1
 * @param molecule2
 * @param atomIndex1
 * @param atomIndex2
 * @param distance
 * @return double
 */
double forceField::correctLinker(const potential::CoulombPotential &coulombPotential,
                                 potential::NonCoulombPotential    &nonCoulombPotential,
                                 physicalData::PhysicalData        &physicalData,
                                 const simulationBox::Molecule     *molecule1,
                                 const simulationBox::Molecule     *molecule2,
                                 const size_t                       atomIndex1,
                                 const size_t                       atomIndex2,
                                 const double                       distance,
                                 const bool                         isDihedral)
{
    const auto chargeProduct = molecule1->getPartialCharge(atomIndex1) * molecule2->getPartialCharge(atomIndex2);

    auto [coulombEnergy, coulombForce] = coulombPotential.calculate(distance, chargeProduct);

    if (isDihedral)
    {
        coulombEnergy *= (1.0 - settings::PotentialSettings::getScale14Coulomb());
        coulombForce  *= (1.0 - settings::PotentialSettings::getScale14Coulomb());
    }

    auto forceMagnitude = -coulombForce;
    physicalData.addCoulombEnergy(-coulombEnergy);

    const auto molType1  = molecule1->getMoltype();
    const auto molType2  = molecule2->getMoltype();
    const auto atomType1 = molecule1->getAtomType(atomIndex1);
    const auto atomType2 = molecule2->getAtomType(atomIndex2);
    const auto vdwType1  = molecule1->getInternalGlobalVDWType(atomIndex1);
    const auto vdwType2  = molecule2->getInternalGlobalVDWType(atomIndex2);

    const auto combinedIndices = {molType1, molType2, atomType1, atomType2, vdwType1, vdwType2};

    if (const auto nonCoulombPair = nonCoulombPotential.getNonCoulombPair(combinedIndices);
        distance < nonCoulombPair->getRadialCutOff())
    {
        auto [nonCoulombEnergy, nonCoulombForce] = nonCoulombPair->calculateEnergyAndForce(distance);

        if (isDihedral)
        {
            nonCoulombEnergy *= (1.0 - settings::PotentialSettings::getScale14VanDerWaals());
            nonCoulombForce  *= (1.0 - settings::PotentialSettings::getScale14VanDerWaals());
        }

        forceMagnitude -= nonCoulombForce;
        physicalData.addNonCoulombEnergy(-nonCoulombEnergy);
    }

    return forceMagnitude;
}