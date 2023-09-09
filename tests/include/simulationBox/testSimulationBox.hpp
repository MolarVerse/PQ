#ifndef _TEST_SIMULATION_BOX_HPP_

#define _TEST_SIMULATION_BOX_HPP_

#include "molecule.hpp"        // for Molecule
#include "simulationBox.hpp"   // for SimulationBox
#include "vector3d.hpp"        // for Vec3D

#include <gtest/gtest.h>   // for Test

class TestSimulationBox : public ::testing::Test
{
  protected:
    virtual void SetUp()
    {
        _simulationBox = new simulationBox::SimulationBox();

        auto molecule1 = simulationBox::Molecule();
        auto molecule2 = simulationBox::Molecule();

        molecule1.setNumberOfAtoms(3);
        molecule2.setNumberOfAtoms(2);
        molecule1.addAtomPosition(linearAlgebra::Vec3D(0.0, 0.0, 0.0));
        molecule1.addAtomPosition(linearAlgebra::Vec3D(1.0, 0.0, 0.0));
        molecule1.addAtomPosition(linearAlgebra::Vec3D(0.0, 1.0, 0.0));
        molecule1.addAtomMass(1.0);
        molecule1.addAtomMass(2.0);
        molecule1.addAtomMass(3.0);
        molecule1.setMolMass(6.0);
        molecule1.setMoltype(1);

        molecule2.addAtomPosition(linearAlgebra::Vec3D(0.0, 0.0, 0.0));
        molecule2.addAtomPosition(linearAlgebra::Vec3D(1.0, 0.0, 0.0));
        molecule2.addAtomMass(1.0);
        molecule2.addAtomMass(2.0);
        molecule2.setMolMass(3.0);
        molecule2.setMoltype(2);

        _simulationBox->addMolecule(molecule1);
        _simulationBox->addMolecule(molecule2);
        _simulationBox->addMoleculeType(molecule1);
        _simulationBox->addMoleculeType(molecule2);

        _simulationBox->setBoxDimensions({10.0, 10.0, 10.0});
    }

    virtual void TearDown() { delete _simulationBox; }

    simulationBox::SimulationBox *_simulationBox;
};

#endif   // _TEST_SIMULATION_BOX_HPP_