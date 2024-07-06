/*****************************************************************************
<GPL_HEADER>

    PQ
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

#include "testPhysicalData.hpp"

#include <memory>   // for allocator

#include "constants/conversionFactors.hpp"   // for _FS_TO_S_
#include "constants/internalConversionFactors.hpp"   // for _KINETIC_ENERGY_FACTOR_, _TEMPERATURE_FACTOR_
#include "gtest/gtest.h"   // for Message, TestPartResult, EXPECT_EQ, TEST_F, Test
#include "staticMatrix.hpp"   // for tensorProduct, diagonalMatrix
#include "vector3d.hpp"       // for operator*, Vector3D

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
    EXPECT_EQ(_physicalData->getMomentum(), linearAlgebra::Vec3D(2.0));
    EXPECT_EQ(_physicalData->getKineticEnergy(), 2.5);
    EXPECT_EQ(_physicalData->getVolume(), 3.0);
    EXPECT_EQ(_physicalData->getDensity(), 3.5);
    EXPECT_EQ(_physicalData->getPressure(), 4.0);
    EXPECT_EQ(_physicalData->getQMEnergy(), 4.5);
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
    EXPECT_EQ(_physicalData->getMomentum(), linearAlgebra::Vec3D(8.0));
    EXPECT_EQ(_physicalData->getKineticEnergy(), 10.0);
    EXPECT_EQ(_physicalData->getVolume(), 12.0);
    EXPECT_EQ(_physicalData->getDensity(), 14.0);
    EXPECT_EQ(_physicalData->getPressure(), 16.0);
    EXPECT_EQ(_physicalData->getQMEnergy(), 18.0);
}

/**
 * @brief tests calculateKinetics function
 *
 */
TEST_F(TestPhysicalData, calculateKinetics)
{
    _physicalData->calculateKinetics(*_simulationBox);

    const auto velocity_mol1_atom1 =
        _simulationBox->getMolecule(0).getAtomVelocity(0);
    const auto velocity_mol1_atom2 =
        _simulationBox->getMolecule(0).getAtomVelocity(1);
    const auto mass_mol1_atom1 = _simulationBox->getMolecule(0).getAtomMass(0);
    const auto mass_mol1_atom2 = _simulationBox->getMolecule(0).getAtomMass(1);

    const auto velocity_mol2_atom1 =
        _simulationBox->getMolecule(1).getAtomVelocity(0);
    const auto mass_mol2_atom1 = _simulationBox->getMolecule(1).getAtomMass(0);

    const auto momentumVector = velocity_mol1_atom1 * mass_mol1_atom1 +
                                velocity_mol1_atom2 * mass_mol1_atom2 +
                                velocity_mol2_atom1 * mass_mol2_atom1;

    const auto kineticEnergyAtomicVector =
        mass_mol1_atom1 * velocity_mol1_atom1 * velocity_mol1_atom1 +
        mass_mol1_atom2 * velocity_mol1_atom2 * velocity_mol1_atom2 +
        mass_mol2_atom1 * velocity_mol2_atom1 * velocity_mol2_atom1;

    const auto kineticEnergyMolecularVector =
        (mass_mol1_atom1 * mass_mol1_atom1 * velocity_mol1_atom1 *
             velocity_mol1_atom1 +
         mass_mol1_atom2 * mass_mol1_atom2 * velocity_mol1_atom2 *
             velocity_mol1_atom2) /
            (mass_mol1_atom1 + mass_mol1_atom2) +
        (mass_mol2_atom1 * mass_mol2_atom1 * velocity_mol2_atom1 *
         velocity_mol2_atom1) /
            mass_mol2_atom1;

    EXPECT_EQ(
        _physicalData->getMomentum(),
        momentumVector * constants::_FS_TO_S_
    );
    EXPECT_EQ(
        diagonal(_physicalData->getKinEnergyAtomTensor()),
        kineticEnergyAtomicVector * constants::_KINETIC_ENERGY_FACTOR_
    );
    EXPECT_EQ(
        diagonal(_physicalData->getKinEnergyMolTensor()),
        kineticEnergyMolecularVector * constants::_KINETIC_ENERGY_FACTOR_
    );
    EXPECT_EQ(
        _physicalData->getKineticEnergy(),
        sum(kineticEnergyAtomicVector) * constants::_KINETIC_ENERGY_FACTOR_
    );
}

/**
 * @brief tests calculateTemperature function
 *
 */
TEST_F(TestPhysicalData, calculateTemperature)
{
    _physicalData->calculateTemperature(*_simulationBox);

    const auto velocity_mol1_atom1 =
        _simulationBox->getMolecule(0).getAtomVelocity(0);
    const auto velocity_mol1_atom2 =
        _simulationBox->getMolecule(0).getAtomVelocity(1);
    const auto mass_mol1_atom1 = _simulationBox->getMolecule(0).getAtomMass(0);
    const auto mass_mol1_atom2 = _simulationBox->getMolecule(0).getAtomMass(1);

    const auto velocity_mol2_atom1 =
        _simulationBox->getMolecule(1).getAtomVelocity(0);
    const auto mass_mol2_atom1 = _simulationBox->getMolecule(1).getAtomMass(0);

    const auto kineticEnergyAtomicVector =
        mass_mol1_atom1 * velocity_mol1_atom1 * velocity_mol1_atom1 +
        mass_mol1_atom2 * velocity_mol1_atom2 * velocity_mol1_atom2 +
        mass_mol2_atom1 * velocity_mol2_atom1 * velocity_mol2_atom1;

    const auto nDOF = _simulationBox->getDegreesOfFreedom();

    EXPECT_NEAR(
        _physicalData->getTemperature(),
        sum(kineticEnergyAtomicVector) * constants::_TEMPERATURE_FACTOR_ /
            (nDOF),
        1e-15
    );
}

/**
 * @brief test reset function
 *
 */
TEST_F(TestPhysicalData, reset)
{
    _physicalData->setKineticEnergy(1.0);
    _physicalData->setCoulombEnergy(1.0);
    _physicalData->setNonCoulombEnergy(1.0);
    _physicalData->setIntraCoulombEnergy(1.0);
    _physicalData->setIntraNonCoulombEnergy(1.0);

    _physicalData->setBondEnergy(1.0);
    _physicalData->setAngleEnergy(1.0);
    _physicalData->setDihedralEnergy(1.0);
    _physicalData->setImproperEnergy(1.0);

    _physicalData->setTemperature(1.0);
    _physicalData->setMomentum(linearAlgebra::Vec3D(1.0));
    _physicalData->setVolume(1.0);
    _physicalData->setDensity(1.0);
    _physicalData->setPressure(1.0);
    _physicalData->setVirial(diagonalMatrix(linearAlgebra::Vec3D(1.0, 1.0, 1.0))
    );
    _physicalData->setQMEnergy(1.0);

    _physicalData->reset();

    EXPECT_EQ(_physicalData->getKineticEnergy(), 0.0);
    EXPECT_EQ(_physicalData->getCoulombEnergy(), 0.0);
    EXPECT_EQ(_physicalData->getNonCoulombEnergy(), 0.0);
    EXPECT_EQ(_physicalData->getIntraCoulombEnergy(), 0.0);
    EXPECT_EQ(_physicalData->getIntraNonCoulombEnergy(), 0.0);

    EXPECT_EQ(_physicalData->getBondEnergy(), 0.0);
    EXPECT_EQ(_physicalData->getAngleEnergy(), 0.0);
    EXPECT_EQ(_physicalData->getDihedralEnergy(), 0.0);
    EXPECT_EQ(_physicalData->getImproperEnergy(), 0.0);

    EXPECT_EQ(_physicalData->getTemperature(), 0.0);
    EXPECT_EQ(_physicalData->getMomentum(), linearAlgebra::Vec3D(0.0));
    EXPECT_EQ(_physicalData->getVolume(), 0.0);
    EXPECT_EQ(_physicalData->getDensity(), 0.0);
    EXPECT_EQ(_physicalData->getPressure(), 0.0);
    EXPECT_EQ(_physicalData->getVirial(), linearAlgebra::tensor3D(0.0));
    EXPECT_EQ(_physicalData->getQMEnergy(), 0.0);
}

/**
 * @brief tests getTotalEnergy function
 *
 */
TEST_F(TestPhysicalData, getTotalEnergy)
{
    _physicalData->setCoulombEnergy(1.0);
    _physicalData->setNonCoulombEnergy(2.0);

    _physicalData->setIntraCoulombEnergy(3.0);
    _physicalData->setIntraNonCoulombEnergy(4.0);

    _physicalData->setBondEnergy(5.0);
    _physicalData->setAngleEnergy(6.0);
    _physicalData->setDihedralEnergy(7.0);
    _physicalData->setImproperEnergy(8.0);

    _physicalData->setKineticEnergy(9.0);
    _physicalData->setQMEnergy(10.0);

    EXPECT_EQ(
        _physicalData->getTotalEnergy(),
        1.0 + 2.0 + 5.0 + 6.0 + 7.0 + 8.0 + 9.0 + 10.0
    );
}

/**
 * @brief tests addIntraCoulombEnergy function
 */
TEST_F(TestPhysicalData, addIntraCoulombEnergy)
{
    _physicalData->setCoulombEnergy(0.0);
    _physicalData->setIntraCoulombEnergy(0.0);
    _physicalData->addIntraCoulombEnergy(1.0);

    EXPECT_EQ(_physicalData->getIntraCoulombEnergy(), 1.0);
    EXPECT_EQ(_physicalData->getCoulombEnergy(), 1.0);
}

/**
 * @brief tests addIntraNonCoulombEnergy function
 */
TEST_F(TestPhysicalData, addIntraNonCoulombEnergy)
{
    _physicalData->setNonCoulombEnergy(0.0);
    _physicalData->setIntraNonCoulombEnergy(0.0);
    _physicalData->addIntraNonCoulombEnergy(1.0);

    EXPECT_EQ(_physicalData->getIntraNonCoulombEnergy(), 1.0);
    EXPECT_EQ(_physicalData->getNonCoulombEnergy(), 1.0);
}