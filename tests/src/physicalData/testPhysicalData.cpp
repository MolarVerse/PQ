#include "testPhysicalData.hpp"

#include "constants.hpp"   // for _KINETIC_ENERGY_FACTOR_, _FS_TO_S_, _TEMPER...
#include "vector3d.hpp"    // for operator*, Vector3D, Vec3D, sum, norm

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <memory>          // for allocator

/**
 * @brief tests makeAverages function
 *
 */
TEST_F(TestPhysicalData, makeAverages)
{
    _physicalData->makeAverages(2);
    EXPECT_EQ(_physicalData->getCoulombEnergy(), 0.5);
    EXPECT_EQ(_physicalData->getNonCoulombEnergy(), 1.0);
    EXPECT_EQ(_physicalData->getTemperature(), 1.5);
    EXPECT_EQ(_physicalData->getMomentum(), 2.0);
    EXPECT_EQ(_physicalData->getKineticEnergy(), 2.5);
    EXPECT_EQ(_physicalData->getVolume(), 3.0);
    EXPECT_EQ(_physicalData->getDensity(), 3.5);
    EXPECT_EQ(_physicalData->getPressure(), 4.0);
}

/**
 * @brief tests updateAverages function
 *
 */
TEST_F(TestPhysicalData, updateAverages)
{
    const physicalData::PhysicalData physicalData2 = *_physicalData;

    _physicalData->updateAverages(physicalData2);
    EXPECT_EQ(_physicalData->getCoulombEnergy(), 2.0);
    EXPECT_EQ(_physicalData->getNonCoulombEnergy(), 4.0);
    EXPECT_EQ(_physicalData->getTemperature(), 6.0);
    EXPECT_EQ(_physicalData->getMomentum(), 8.0);
    EXPECT_EQ(_physicalData->getKineticEnergy(), 10.0);
    EXPECT_EQ(_physicalData->getVolume(), 12.0);
    EXPECT_EQ(_physicalData->getDensity(), 14.0);
    EXPECT_EQ(_physicalData->getPressure(), 16.0);
}

/**
 * @brief tests calculateKineticEnergyAndMomentum function
 *
 */
TEST_F(TestPhysicalData, calculateKineticEnergyAndMomentum)
{
    _physicalData->calculateKineticEnergyAndMomentum(*_simulationBox);

    const auto velocity_mol1_atom1 = _simulationBox->getMolecule(0).getAtomVelocity(0);
    const auto velocity_mol1_atom2 = _simulationBox->getMolecule(0).getAtomVelocity(1);
    const auto mass_mol1_atom1     = _simulationBox->getMolecule(0).getAtomMass(0);
    const auto mass_mol1_atom2     = _simulationBox->getMolecule(0).getAtomMass(1);

    const auto velocity_mol2_atom1 = _simulationBox->getMolecule(1).getAtomVelocity(0);
    const auto mass_mol2_atom1     = _simulationBox->getMolecule(1).getAtomMass(0);

    const auto momentumVector =
        velocity_mol1_atom1 * mass_mol1_atom1 + velocity_mol1_atom2 * mass_mol1_atom2 + velocity_mol2_atom1 * mass_mol2_atom1;

    const auto kineticEnergyAtomicVector = mass_mol1_atom1 * velocity_mol1_atom1 * velocity_mol1_atom1 +
                                           mass_mol1_atom2 * velocity_mol1_atom2 * velocity_mol1_atom2 +
                                           mass_mol2_atom1 * velocity_mol2_atom1 * velocity_mol2_atom1;

    const auto kineticEnergyMolecularVector =
        (mass_mol1_atom1 * mass_mol1_atom1 * velocity_mol1_atom1 * velocity_mol1_atom1 +
         mass_mol1_atom2 * mass_mol1_atom2 * velocity_mol1_atom2 * velocity_mol1_atom2) /
            (mass_mol1_atom1 + mass_mol1_atom2) +
        (mass_mol2_atom1 * mass_mol2_atom1 * velocity_mol2_atom1 * velocity_mol2_atom1) / mass_mol2_atom1;

    EXPECT_EQ(_physicalData->getMomentumVector(), momentumVector * constants::_FS_TO_S_);
    EXPECT_EQ(_physicalData->getMomentum(), norm(momentumVector) * constants::_FS_TO_S_);
    EXPECT_EQ(_physicalData->getKineticEnergyAtomicVector(), kineticEnergyAtomicVector * constants::_KINETIC_ENERGY_FACTOR_);
    EXPECT_EQ(_physicalData->getKineticEnergyMolecularVector(),
              kineticEnergyMolecularVector * constants::_KINETIC_ENERGY_FACTOR_);
    EXPECT_EQ(_physicalData->getKineticEnergy(), sum(kineticEnergyAtomicVector) * constants::_KINETIC_ENERGY_FACTOR_);
}

/**
 * @brief tests calculateTemperature function
 *
 */
TEST_F(TestPhysicalData, calculateTemperature)
{
    _physicalData->calculateTemperature(*_simulationBox);

    const auto velocity_mol1_atom1 = _simulationBox->getMolecule(0).getAtomVelocity(0);
    const auto velocity_mol1_atom2 = _simulationBox->getMolecule(0).getAtomVelocity(1);
    const auto mass_mol1_atom1     = _simulationBox->getMolecule(0).getAtomMass(0);
    const auto mass_mol1_atom2     = _simulationBox->getMolecule(0).getAtomMass(1);

    const auto velocity_mol2_atom1 = _simulationBox->getMolecule(1).getAtomVelocity(0);
    const auto mass_mol2_atom1     = _simulationBox->getMolecule(1).getAtomMass(0);

    const auto kineticEnergyAtomicVector = mass_mol1_atom1 * velocity_mol1_atom1 * velocity_mol1_atom1 +
                                           mass_mol1_atom2 * velocity_mol1_atom2 * velocity_mol1_atom2 +
                                           mass_mol2_atom1 * velocity_mol2_atom1 * velocity_mol2_atom1;

    const auto nDOF = _simulationBox->getDegreesOfFreedom();

    EXPECT_EQ(_physicalData->getTemperature(), sum(kineticEnergyAtomicVector) * constants::_TEMPERATURE_FACTOR_ / (nDOF));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}