#ifndef _TEST_THERMOSTAT_HPP_

#define _TEST_THERMOSTAT_HPP_

#include "physicalData.hpp"    // for PhysicalData
#include "simulationBox.hpp"   // for SimulationBox
#include "thermostat.hpp"

#include <gtest/gtest.h>

class TestThermostat : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        _thermostat = new thermostat::Thermostat();
        _data       = new physicalData::PhysicalData();

        _simulationBox = new simulationBox::SimulationBox();

        auto molecule1 = simulationBox::Molecule();
        molecule1.setNumberOfAtoms(2);
        molecule1.addAtomMass(1.0);
        molecule1.addAtomMass(1.0);
        molecule1.setMolMass(2.0);
        molecule1.addAtomVelocity(linearAlgebra::Vec3D(1.0, 1.0, 1.0));
        molecule1.addAtomVelocity(linearAlgebra::Vec3D(1.0, 2.0, 3.0));

        auto molecule2 = simulationBox::Molecule();
        molecule2.setNumberOfAtoms(1);
        molecule2.addAtomMass(1.0);
        molecule2.setMolMass(1.0);
        molecule2.addAtomVelocity(linearAlgebra::Vec3D(1.0, 1.0, 1.0));

        _simulationBox->addMolecule(molecule1);
        _simulationBox->addMolecule(molecule2);

        _simulationBox->calculateDegreesOfFreedom();
    }

    void TearDown() override
    {
        delete _data;
        delete _simulationBox;
        delete _thermostat;
    }

    physicalData::PhysicalData   *_data;
    simulationBox::SimulationBox *_simulationBox;
    thermostat::Thermostat       *_thermostat;
};

#endif