#ifndef _TEST_RESET_KINETICS_HPP_

#define _TEST_RESET_KINETICS_HPP_

#include "resetKinetics.hpp"

#include <gtest/gtest.h>

class TestResetKinetics : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        _resetKinetics = new resetKinetics::ResetKinetics(1, 2, 3, 4, 300.0);
        _data          = new physicalData::PhysicalData();

        _simulationBox = new simulationBox::SimulationBox();

        auto molecule1 = simulationBox::Molecule();
        molecule1.setNumberOfAtoms(2);
        molecule1.addAtomVelocity(linearAlgebra::Vec3D(1.0, 1.0, 1.0));
        molecule1.addAtomVelocity(linearAlgebra::Vec3D(1.0, 2.0, 3.0));
        molecule1.addAtomMass(1.0);
        molecule1.addAtomMass(1.0);

        auto molecule2 = simulationBox::Molecule();
        molecule2.setNumberOfAtoms(1);
        molecule2.addAtomVelocity(linearAlgebra::Vec3D(1.0, 1.0, 1.0));
        molecule2.addAtomMass(1.0);

        _simulationBox->addMolecule(molecule1);
        _simulationBox->addMolecule(molecule2);
        _simulationBox->setTotalMass(3.0);
    }

    void TearDown() override
    {
        delete _data;
        delete _simulationBox;
        delete _resetKinetics;
    }

    physicalData::PhysicalData   *_data;
    simulationBox::SimulationBox *_simulationBox;
    resetKinetics::ResetKinetics *_resetKinetics;
};

#endif