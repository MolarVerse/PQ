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

#include <cmath>   // for sqrt

#include "berendsenThermostat.hpp"                   // for BerendsenThermostat
#include "constants/internalConversionFactors.hpp"   // for _TEMPERATURE_FACTOR_
#include "gtest/gtest.h"                             // for InitGoogleTest
#include "langevinThermostat.hpp"                    // for LangevinThermostat
#include "noseHooverThermostat.hpp"                  // for NoseHooverThermostat
#include "physicalData.hpp"                          // for PhysicalData
#include "simulationBox.hpp"                         // for SimulationBox
#include "thermostatSettings.hpp"                    // for ThermostatType
#include "timingsSettings.hpp"                       // for TimingsSettings
#include "velocityRescalingThermostat.hpp"   // for VelocityRescalingThermostat

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
        sum(kineticEnergyAtomicVector) * constants::TEMPERATURE_FACTOR / (nDOF)
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
                                constants::TEMPERATURE_FACTOR /
                                static_cast<double>(nDOF);

    const auto berendsenFactor =
        ::sqrt(1.0 + 0.1 / 100.0 * (300.0 / oldTemperature - 1.0));

    _thermostat->applyThermostat(*_simulationBox, *_data);

    EXPECT_EQ(
        _data->getTemperature(),
        oldTemperature * berendsenFactor * berendsenFactor
    );
}

/* ---------- VelocityRescalingThermostat ---------- */

TEST_F(TestThermostat, velocityRescaling_tauSetterGetter)
{
    auto vr = thermostat::VelocityRescalingThermostat(300.0, 100.0);
    EXPECT_DOUBLE_EQ(vr.getTau(), 100.0);

    vr.setTau(50.0);
    EXPECT_DOUBLE_EQ(vr.getTau(), 50.0);
}

TEST_F(TestThermostat, velocityRescaling_thermostatType)
{
    auto vr = thermostat::VelocityRescalingThermostat(300.0, 100.0);
    EXPECT_EQ(
        vr.getThermostatType(),
        settings::ThermostatType::VELOCITY_RESCALING
    );
}

TEST_F(TestThermostat, velocityRescaling_applyDoesNotNaN)
{
    delete _thermostat;
    _thermostat = new thermostat::VelocityRescalingThermostat(300.0, 100.0);
    settings::TimingsSettings::setTimeStep(0.1);

    _thermostat->applyThermostat(*_simulationBox, *_data);

    EXPECT_FALSE(std::isnan(_data->getTemperature()));
    EXPECT_FALSE(std::isinf(_data->getTemperature()));
    for (const auto &atom : _simulationBox->getAtoms())
        for (size_t i = 0; i < 3; ++i)
        {
            EXPECT_FALSE(std::isnan(atom->getVelocity()[i]));
            EXPECT_FALSE(std::isinf(atom->getVelocity()[i]));
        }
}

// Regression test: starting from zero kinetic energy (T == 0) used to
// produce NaN velocities, because tempRatio = T_target / 0 = Inf and
// the velocity scaling 0 * Inf = NaN. The guard skips the scaling and
// leaves velocities at zero.
TEST_F(TestThermostat, applyBerendsen_zeroTemperatureNoNaN)
{
    delete _thermostat;
    _thermostat = new thermostat::BerendsenThermostat(300.0, 100.0);
    settings::TimingsSettings::setTimeStep(0.1);

    for (auto &atom : _simulationBox->getAtoms())
        atom->setVelocity({0.0, 0.0, 0.0});

    _thermostat->applyThermostat(*_simulationBox, *_data);

    EXPECT_FALSE(std::isnan(_data->getTemperature()));
    EXPECT_FALSE(std::isinf(_data->getTemperature()));
    for (const auto &atom : _simulationBox->getAtoms())
        for (size_t i = 0; i < 3; ++i)
        {
            EXPECT_FALSE(std::isnan(atom->getVelocity()[i]));
            EXPECT_FALSE(std::isinf(atom->getVelocity()[i]));
        }
}

/* ---------- LangevinThermostat ---------- */

TEST_F(TestThermostat, langevin_constructorComputesSigma)
{
    // sigma > 0 once friction and targetTemp are non-zero.
    const auto langevin = thermostat::LangevinThermostat(300.0, 0.1);
    EXPECT_GT(langevin.getSigma(), 0.0);
    EXPECT_DOUBLE_EQ(langevin.getFriction(), 0.1);
}

TEST_F(TestThermostat, langevin_zeroFrictionHasZeroSigma)
{
    const auto langevin = thermostat::LangevinThermostat(300.0, 0.0);

    EXPECT_DOUBLE_EQ(langevin.getSigma(), 0.0);
    EXPECT_DOUBLE_EQ(langevin.getFriction(), 0.0);
}

TEST_F(TestThermostat, langevin_settersAndGetters)
{
    auto langevin = thermostat::LangevinThermostat(300.0, 0.1);

    langevin.setFriction(0.5);
    EXPECT_DOUBLE_EQ(langevin.getFriction(), 0.5);

    langevin.setSigma(2.0);
    EXPECT_DOUBLE_EQ(langevin.getSigma(), 2.0);
}

TEST_F(TestThermostat, langevin_setTargetTemperatureRecomputesSigma)
{
    auto langevin = thermostat::LangevinThermostat(300.0, 0.1);
    settings::TimingsSettings::setTimeStep(0.1);
    const auto sigmaAt300 = langevin.getSigma();

    langevin.setTargetTemperature(600.0);
    const auto sigmaAt600 = langevin.getSigma();

    // Setting a higher target temperature must update sigma (the
    // Langevin Gaussian-noise amplitude scales with sqrt(kBT)).
    EXPECT_NE(sigmaAt300, sigmaAt600);
    EXPECT_GT(sigmaAt600, sigmaAt300);
}

TEST_F(TestThermostat, langevin_setFrictionRecomputesSigma)
{
    auto langevin = thermostat::LangevinThermostat(300.0, 0.1);
    settings::TimingsSettings::setTimeStep(0.1);
    const auto sigmaAtFrictionPointOne = langevin.getSigma();

    langevin.setFriction(0.5);
    const auto sigmaAtFrictionPointFive = langevin.getSigma();

    EXPECT_NE(sigmaAtFrictionPointOne, sigmaAtFrictionPointFive);
    EXPECT_GT(sigmaAtFrictionPointFive, sigmaAtFrictionPointOne);
}

TEST_F(TestThermostat, langevin_thermostatType)
{
    auto langevin = thermostat::LangevinThermostat(300.0, 0.1);
    EXPECT_EQ(langevin.getThermostatType(), settings::ThermostatType::LANGEVIN);
}

/* ---------- NoseHooverThermostat ---------- */

TEST_F(TestThermostat, noseHoover_thermostatType)
{
    auto nh = thermostat::NoseHooverThermostat(
        300.0,
        std::vector<double>{0.0, 0.0, 0.0},
        std::vector<double>{0.0, 0.0, 0.0},
        1.0e13
    );
    EXPECT_EQ(nh.getThermostatType(), settings::ThermostatType::NOSE_HOOVER);
}

TEST_F(TestThermostat, noseHoover_couplingFrequencySetterGetter)
{
    auto nh = thermostat::NoseHooverThermostat(
        300.0,
        std::vector<double>{0.0, 0.0, 0.0},
        std::vector<double>{0.0, 0.0, 0.0},
        1.0e13
    );
    EXPECT_DOUBLE_EQ(nh.getCouplingFrequency(), 1.0e13);

    nh.setCouplingFrequency(5.0e12);
    EXPECT_DOUBLE_EQ(nh.getCouplingFrequency(), 5.0e12);
}

TEST_F(TestThermostat, noseHoover_setChiAtIndex)
{
    auto nh = thermostat::NoseHooverThermostat(
        300.0,
        std::vector<double>{0.0, 0.0, 0.0},
        std::vector<double>{0.0, 0.0, 0.0},
        1.0e13
    );
    nh.setChi(2u, 7.0);
    EXPECT_DOUBLE_EQ(nh.getChi()[2], 7.0);

    nh.setZeta(1u, 3.0);
    EXPECT_DOUBLE_EQ(nh.getZeta()[1], 3.0);
}

TEST_F(TestThermostat, noseHoover_appliesFiniteForceAndStateUpdates)
{
    auto nh = thermostat::NoseHooverThermostat(
        300.0,
        std::vector<double>{0.1, 0.2, 0.3},
        std::vector<double>{0.0, 0.0, 0.0},
        1.0
    );
    settings::TimingsSettings::setTimeStep(0.1);

    nh.applyThermostatOnForces(*_simulationBox);
    for (const auto &atom : _simulationBox->getAtoms())
        for (size_t axis = 0; axis < 3; ++axis)
            EXPECT_TRUE(std::isfinite(atom->getForce()[axis]));

    const auto chiBefore  = nh.getChi();
    const auto zetaBefore = nh.getZeta();
    nh.applyThermostat(*_simulationBox, *_data);

    EXPECT_TRUE(std::isfinite(_data->getTemperature()));
    EXPECT_TRUE(std::isfinite(_data->getNoseHooverMomentumEnergy()));
    EXPECT_TRUE(std::isfinite(_data->getNoseHooverFrictionEnergy()));
    EXPECT_NE(nh.getChi(), chiBefore);
    EXPECT_NE(nh.getZeta(), zetaBefore);
}
