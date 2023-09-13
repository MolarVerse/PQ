#ifndef _TEST_PHYSICAL_DATA_HPP_

#define _TEST_PHYSICAL_DATA_HPP_

#include "atom.hpp"            // for Atom
#include "molecule.hpp"        // for Molecule
#include "physicalData.hpp"    // for PhysicalData
#include "simulationBox.hpp"   // for SimulationBox
#include "vector3d.hpp"        // for Vec3D

#include <gtest/gtest.h>   // for Test
#include <memory>          // for make_shared, __shared_ptr_access, shared_ptr

/**
 * @class TestPhysicalData
 *
 * @brief test suite for physical data
 *
 */
class TestPhysicalData : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        _physicalData = new physicalData::PhysicalData();
        _physicalData->setCoulombEnergy(1.0);
        _physicalData->setNonCoulombEnergy(2.0);
        _physicalData->setTemperature(3.0);
        _physicalData->setMomentum(4.0);
        _physicalData->setKineticEnergy(5.0);
        _physicalData->setVolume(6.0);
        _physicalData->setDensity(7.0);
        _physicalData->setPressure(8.0);
        _physicalData->setQMEnergy(9.0);

        _simulationBox = new simulationBox::SimulationBox();

        auto molecule1 = simulationBox::Molecule();

        auto atom1 = std::make_shared<simulationBox::Atom>();
        auto atom2 = std::make_shared<simulationBox::Atom>();

        molecule1.setNumberOfAtoms(2);

        atom1->setMass(1.0);
        atom2->setMass(1.0);
        atom1->setVelocity(linearAlgebra::Vec3D(1.0, 1.0, 1.0));
        atom2->setVelocity(linearAlgebra::Vec3D(1.0, 2.0, 3.0));
        molecule1.setMolMass(2.0);
        molecule1.addAtom(atom1);
        molecule1.addAtom(atom2);

        auto molecule2 = simulationBox::Molecule();
        molecule2.setNumberOfAtoms(1);

        auto atom3 = std::make_shared<simulationBox::Atom>();
        atom3->setMass(1.0);
        atom3->setVelocity(linearAlgebra::Vec3D(1.0, 1.0, 1.0));
        molecule2.setMolMass(1.0);
        molecule2.addAtom(atom3);

        _simulationBox->addMolecule(molecule1);
        _simulationBox->addMolecule(molecule2);

        _simulationBox->calculateDegreesOfFreedom();
    }
    void TearDown() override { delete _physicalData; }

    physicalData::PhysicalData   *_physicalData;
    simulationBox::SimulationBox *_simulationBox;
};

#endif