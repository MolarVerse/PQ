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

#include "testThermostat.hpp"

#include <cmath>    // for sqrt
#include <memory>   // for allocator

#include "berendsenThermostat.hpp"                   // for BerendsenThermostat
#include "constants/internalConversionFactors.hpp"   // for _TEMPERATURE_FACTOR_
#include "gtest/gtest.h"                             // for InitGoogleTest
#include "noseHooverThermostat.hpp"                  // for NoseHooverThermostat
#include "physicalData.hpp"                          // for PhysicalData
#include "simulationBox.hpp"                         // for SimulationBox
#include "timingsSettings.hpp"                       // for TimingsSettings

TEST_F(TestThermostat, calculateTemperature)
{
    _thermostat->applyThermostat(*_simulationBox, *_data);

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

    EXPECT_EQ(
        _data->getTemperature(),
        sum(kineticEnergyAtomicVector) * constants::_TEMPERATURE_FACTOR_ /
            (nDOF)
    );
}

TEST_F(TestThermostat, applyTemperatureRamping)
{
    _thermostat->setTemperatureIncrease(0.0);
    _thermostat->setTemperatureRampingSteps(0);
    _thermostat->setTemperatureRampingFrequency(1);
    _thermostat->setTargetTemperature(300.0);

    _thermostat->applyTemperatureRamping();
    EXPECT_EQ(_thermostat->getTargetTemperature(), 300.0);

    _thermostat->setTemperatureIncrease(1.0);
    _thermostat->setTemperatureRampingSteps(1);
    _thermostat->setTemperatureRampingFrequency(1);

    _thermostat->applyTemperatureRamping();
    EXPECT_EQ(_thermostat->getTargetTemperature(), 301.0);

    _thermostat->applyTemperatureRamping();
    EXPECT_EQ(_thermostat->getTargetTemperature(), 301.0);

    _thermostat->setTemperatureIncrease(1.0);
    _thermostat->setTemperatureRampingSteps(2);
    _thermostat->setTemperatureRampingFrequency(1);
    _thermostat->setTargetTemperature(300.0);

    _thermostat->applyTemperatureRamping();
    EXPECT_EQ(_thermostat->getTargetTemperature(), 301.0);

    _thermostat->applyTemperatureRamping();
    EXPECT_EQ(_thermostat->getTargetTemperature(), 302.0);

    _thermostat->applyTemperatureRamping();
    EXPECT_EQ(_thermostat->getTargetTemperature(), 302.0);

    _thermostat->setTemperatureIncrease(1.0);
    _thermostat->setTemperatureRampingSteps(4);
    _thermostat->setTemperatureRampingFrequency(2);
    _thermostat->setTargetTemperature(300.0);

    _thermostat->applyTemperatureRamping();
    EXPECT_EQ(_thermostat->getTargetTemperature(), 300.0);

    _thermostat->applyTemperatureRamping();
    EXPECT_EQ(_thermostat->getTargetTemperature(), 301.0);

    _thermostat->applyTemperatureRamping();
    EXPECT_EQ(_thermostat->getTargetTemperature(), 301.0);

    _thermostat->applyTemperatureRamping();
    EXPECT_EQ(_thermostat->getTargetTemperature(), 302.0);

    _thermostat->applyTemperatureRamping();
    EXPECT_EQ(_thermostat->getTargetTemperature(), 302.0);

    _thermostat->applyTemperatureRamping();
    EXPECT_EQ(_thermostat->getTargetTemperature(), 302.0);
}

TEST_F(TestThermostat, applyThermostatBerendsen)
{
    _thermostat = new thermostat::BerendsenThermostat(300.0, 100.0);
    settings::TimingsSettings::setTimeStep(0.1);

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

    const auto oldTemperature = sum(kineticEnergyAtomicVector) *
                                constants::_TEMPERATURE_FACTOR_ /
                                static_cast<double>(nDOF);

    const auto berendsenFactor =
        ::sqrt(1.0 + 0.1 / 100.0 * (300.0 / oldTemperature - 1.0));

    _thermostat->applyThermostat(*_simulationBox, *_data);

    EXPECT_EQ(
        _data->getTemperature(),
        oldTemperature * berendsenFactor * berendsenFactor
    );
}

// Regression test: the Nose-Hoover chain velocity-update loop used to
// be bounded by `i < _chi.size() - 1`, so the last chain element was
// never evolved during applyThermostat. With a chain of size 3 starting
// at chi = zeta = 0, the previously-frozen _zeta[2] should accumulate a
// non-zero value after a single applyThermostat call (driven by the
// momentum term from chi[1]).
TEST_F(TestThermostat, applyThermostatNoseHoover_lastChainElementEvolves)
{
    delete _thermostat;
    auto *nh = new thermostat::NoseHooverThermostat(
        300.0,                                       // targetTemp
        std::vector<double>{0.0, 0.0, 0.0},          // chi (all zero)
        std::vector<double>{0.0, 0.0, 0.0},          // zeta (all zero)
        1.0e13                                       // couplingFrequency (1/s)
    );
    _thermostat = nh;
    settings::TimingsSettings::setTimeStep(0.5);

    _data->calculateTemperature(*_simulationBox);
    _thermostat->applyThermostat(*_simulationBox, *_data);

    const auto chi  = nh->getChi();
    const auto zeta = nh->getZeta();

    // The previously-frozen last chain element must now accumulate a
    // (small) non-zero zeta value from the chi[1] momentum term.
    EXPECT_NE(zeta[2], 0.0) << "last zeta did not evolve";
}
