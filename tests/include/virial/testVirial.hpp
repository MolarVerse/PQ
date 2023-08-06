#ifndef _TEST_VIRIAL_HPP_

#define _TEST_VIRIAL_HPP_

#include "virial.hpp"

#include <gtest/gtest.h>

class TestVirial : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        _virial = new virial::Virial();
        _data   = new physicalData::PhysicalData();

        _simulationBox = new simulationBox::SimulationBox();

        auto molecule1 = simulationBox::Molecule();
        molecule1.setNumberOfAtoms(2);
        molecule1.addAtomPosition(linearAlgebra::Vec3D(1.0, 1.0, 1.0));
        molecule1.addAtomPosition(linearAlgebra::Vec3D(1.0, 2.0, 3.0));
        molecule1.addAtomForce(linearAlgebra::Vec3D(1.0, 1.0, 1.0));
        molecule1.addAtomForce(linearAlgebra::Vec3D(1.0, 2.0, 3.0));
        molecule1.resizeAtomShiftForces();
        molecule1.setAtomShiftForces(0, linearAlgebra::Vec3D(1.0, 1.0, 1.0));
        molecule1.setAtomShiftForces(1, linearAlgebra::Vec3D(1.0, 2.0, 3.0));
        molecule1.setCenterOfMass(linearAlgebra::Vec3D(1.0, 1.0, 1.0));

        auto molecule2 = simulationBox::Molecule();
        molecule2.setNumberOfAtoms(1);
        molecule2.addAtomPosition(linearAlgebra::Vec3D(1.0, 1.0, 1.0));
        molecule2.addAtomForce(linearAlgebra::Vec3D(1.0, 1.0, 1.0));
        molecule2.resizeAtomShiftForces();
        molecule2.setAtomShiftForces(0, linearAlgebra::Vec3D(1.0, 1.0, 1.0));
        molecule2.setCenterOfMass(linearAlgebra::Vec3D(0.0, 0.0, 0.0));

        _simulationBox->addMolecule(molecule1);
        _simulationBox->addMolecule(molecule2);

        _simulationBox->setBoxDimensions(linearAlgebra::Vec3D(10.0, 10.0, 10.0));
    }

    void TearDown() override
    {
        delete _data;
        delete _simulationBox;
        delete _virial;
    }

    physicalData::PhysicalData   *_data;
    simulationBox::SimulationBox *_simulationBox;
    virial::Virial               *_virial;
};

#endif