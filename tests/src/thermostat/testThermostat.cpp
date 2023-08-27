#include "testThermostat.hpp"

#include "constants.hpp"       // for _TEMPERATURE_FACTOR_
#include "physicalData.hpp"    // for PhysicalData
#include "simulationBox.hpp"   // for SimulationBox

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <math.h>          // for sqrt
#include <memory>          // for allocator

TEST_F(TestThermostat, calculateTemperature)
{
    _thermostat->applyThermostat(*_simulationBox, *_data);

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

    EXPECT_EQ(_data->getTemperature(), sum(kineticEnergyAtomicVector) * constants::_TEMPERATURE_FACTOR_ / (nDOF));
}

TEST_F(TestThermostat, applyThermostatBerendsen)
{
    _thermostat = new thermostat::BerendsenThermostat(300.0, 100.0);
    _thermostat->setTimestep(0.1);

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

    const auto oldTemperature = sum(kineticEnergyAtomicVector) * constants::_TEMPERATURE_FACTOR_ / static_cast<double>(nDOF);

    const auto berendsenFactor = ::sqrt(1.0 + 0.1 / 100.0 * (300.0 / oldTemperature - 1.0));

    _thermostat->applyThermostat(*_simulationBox, *_data);

    EXPECT_EQ(_data->getTemperature(), oldTemperature * berendsenFactor * berendsenFactor);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}