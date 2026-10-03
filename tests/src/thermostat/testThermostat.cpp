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

#include <gtest/gtest.h>

#include <cmath>   // for sqrt
#include <memory>

#include "berendsenThermostat.hpp"                   // for BerendsenThermostat
#include "constants/internalConversionFactors.hpp"   // for _TEMPERATURE_FACTOR_
#include "exceptions.hpp"                            // for UserInputException
#include "langevinThermostat.hpp"                    // for LangevinThermostat
#include "noseHooverThermostat.hpp"                  // for NoseHooverThermostat
#include "physicalData.hpp"                          // for PhysicalData
#include "simulationBox.hpp"                         // for SimulationBox
#include "throwWithMessage.hpp"
#include "timingsSettings.hpp"               // for TimingsSettings
#include "velocityRescalingThermostat.hpp"   // for VelocityRescalingThermostat

TEST_F(TestThermostat, calculateTemperature)
{
    _thermostat->applyThermostat(*_simulationBox, *_data);

    const auto mol1                = _simulationBox->getMolecule(0);
    const auto velocity_mol1_atom1 = mol1.getAtomVelocity(AtomIndex{0});
    const auto velocity_mol1_atom2 = mol1.getAtomVelocity(AtomIndex{1});
    const auto mass_mol1_atom1     = mol1.getAtomMass(AtomIndex{0});
    const auto mass_mol1_atom2     = mol1.getAtomMass(AtomIndex{1});

    const auto mol2                = _simulationBox->getMolecule(1);
    const auto velocity_mol2_atom1 = mol2.getAtomVelocity(AtomIndex{0});
    const auto mass_mol2_atom1     = mol2.getAtomMass(AtomIndex{0});

    const auto kineticEnergyAtomicVector =
        mass_mol1_atom1 * velocity_mol1_atom1 * velocity_mol1_atom1 +
        mass_mol1_atom2 * velocity_mol1_atom2 * velocity_mol1_atom2 +
        mass_mol2_atom1 * velocity_mol2_atom1 * velocity_mol2_atom1;

    const auto nDOF = _simulationBox->getDegreesOfFreedom();

    EXPECT_EQ(
        _data->getTemperature(),
        sum(kineticEnergyAtomicVector) * TEMPERATURE_FACTOR /
            static_cast<double>(nDOF)
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
    _thermostat =
        std::make_unique<thermostat::BerendsenThermostat>(300.0, 100.0);
    settings::TimingsSettings::setTimeStep(0.1);

    const auto mol1                = _simulationBox->getMolecule(0);
    const auto velocity_mol1_atom1 = mol1.getAtomVelocity(AtomIndex{0});
    const auto velocity_mol1_atom2 = mol1.getAtomVelocity(AtomIndex{1});
    const auto mass_mol1_atom1     = mol1.getAtomMass(AtomIndex{0});
    const auto mass_mol1_atom2     = mol1.getAtomMass(AtomIndex{1});

    const auto mol2                = _simulationBox->getMolecule(1);
    const auto velocity_mol2_atom1 = mol2.getAtomVelocity(AtomIndex{0});
    const auto mass_mol2_atom1     = mol2.getAtomMass(AtomIndex{0});

    const auto kineticEnergyAtomicVector =
        mass_mol1_atom1 * velocity_mol1_atom1 * velocity_mol1_atom1 +
        mass_mol1_atom2 * velocity_mol1_atom2 * velocity_mol1_atom2 +
        mass_mol2_atom1 * velocity_mol2_atom1 * velocity_mol2_atom1;

    const auto nDOF = _simulationBox->getDegreesOfFreedom();

    const auto oldTemperature = sum(kineticEnergyAtomicVector) *
                                TEMPERATURE_FACTOR / static_cast<double>(nDOF);

    const auto berendsenFactor =
        ::sqrt(1.0 + (0.1 / 100.0 * (300.0 / oldTemperature - 1.0)));

    _thermostat->applyThermostat(*_simulationBox, *_data);

    EXPECT_EQ(
        _data->getTemperature(),
        oldTemperature * berendsenFactor * berendsenFactor
    );
}

/* ---------- VelocityRescalingThermostat ---------- */

TEST_F(TestThermostat, velocityRescalingTauSetterGetter)
{
    auto thermostat = thermostat::VelocityRescalingThermostat(300.0, 100.0);
    EXPECT_DOUBLE_EQ(thermostat.getTau(), 100.0);

    thermostat.setTau(50.0);
    EXPECT_DOUBLE_EQ(thermostat.getTau(), 50.0);
}

TEST_F(TestThermostat, velocityRescalingThermostatType)
{
    auto thermostat = thermostat::VelocityRescalingThermostat(300.0, 100.0);
    EXPECT_EQ(
        thermostat.getThermostatType(),
        ThermostatType::VELOCITY_RESCALING
    );
}

TEST_F(TestThermostat, velocityRescalingApplyDoesNotNaN)
{
    _thermostat =
        std::make_unique<thermostat::VelocityRescalingThermostat>(300.0, 100.0);
    settings::TimingsSettings::setTimeStep(0.1);

    _thermostat->applyThermostat(*_simulationBox, *_data);

    EXPECT_FALSE(std::isnan(_data->getTemperature()));
    EXPECT_FALSE(std::isinf(_data->getTemperature()));
    for (const auto &atom : _simulationBox->getAtoms())
    {
        for (size_t i = 0; i < 3; ++i)
        {
            EXPECT_FALSE(std::isnan(atom->getVelocity()[i]));
            EXPECT_FALSE(std::isinf(atom->getVelocity()[i]));
        }
    }
}

TEST_F(TestThermostat, berendsenZeroTemperatureDoesNotNaN)
{
    _thermostat = std::make_unique<thermostat::BerendsenThermostat>(0.0, 100.0);
    settings::TimingsSettings::setTimeStep(0.1);

    for (auto &atom : _simulationBox->getAtoms())
        atom->setVelocity({0.0, 0.0, 0.0});

    _thermostat->applyThermostat(*_simulationBox, *_data);

    EXPECT_TRUE(std::isfinite(_data->getTemperature()));
    for (const auto &atom : _simulationBox->getAtoms())
        for (size_t dimension = 0; dimension < 3; ++dimension)
            EXPECT_TRUE(std::isfinite(atom->getVelocity()[dimension]));
}

TEST_F(TestThermostat, berendsenRejectsPositiveTargetFromZero)
{
    _thermostat =
        std::make_unique<thermostat::BerendsenThermostat>(300.0, 100.0);
    settings::TimingsSettings::setTimeStep(0.1);

    for (auto &atom : _simulationBox->getAtoms())
        atom->setVelocity({0.0, 0.0, 0.0});

    EXPECT_THROW_MSG(
        _thermostat->applyThermostat(*_simulationBox, *_data),
        exc::UserInputException,
        "Cannot apply Berendsen coupling to a zero-temperature system with a "
        "positive target temperature. Initialize velocities first."
    );
}

TEST_F(TestThermostat, velocityRescalingZeroTemperatureDoesNotNaN)
{
    _thermostat =
        std::make_unique<thermostat::VelocityRescalingThermostat>(0.0, 100.0);
    settings::TimingsSettings::setTimeStep(0.1);

    for (auto &atom : _simulationBox->getAtoms())
        atom->setVelocity({0.0, 0.0, 0.0});

    _thermostat->applyThermostat(*_simulationBox, *_data);

    EXPECT_TRUE(std::isfinite(_data->getTemperature()));
    for (const auto &atom : _simulationBox->getAtoms())
        for (size_t dimension = 0; dimension < 3; ++dimension)
            EXPECT_TRUE(std::isfinite(atom->getVelocity()[dimension]));
}

TEST_F(TestThermostat, velocityRescalingRejectsPositiveTargetFromZero)
{
    _thermostat =
        std::make_unique<thermostat::VelocityRescalingThermostat>(300.0, 100.0);
    settings::TimingsSettings::setTimeStep(0.1);

    for (auto &atom : _simulationBox->getAtoms())
        atom->setVelocity({0.0, 0.0, 0.0});

    EXPECT_THROW_MSG(
        _thermostat->applyThermostat(*_simulationBox, *_data),
        exc::UserInputException,
        "Cannot apply velocity rescaling to a zero-temperature system with a "
        "positive target temperature. Initialize velocities first."
    );
}

namespace
{
    bool velocitiesAreFinite(const molsys::SimulationBox &box)
    {
        for (const auto &atom : box.getAtoms())
            for (size_t axis = 0; axis < 3; ++axis)
                if (!std::isfinite(atom->getVelocity()[axis]))
                    return false;
        return true;
    }
}   // namespace

TEST_F(TestThermostat, berendsenRejectsRelaxationTimeShorterThanTimestep)
{
    _data->calculateTemperature(*_simulationBox);
    _thermostat = std::make_unique<thermostat::BerendsenThermostat>(
        0.5 * _data->getTemperature(),
        0.01
    );
    settings::TimingsSettings::setTimeStep(0.1);

    EXPECT_THROW_MSG(
        _thermostat->applyThermostat(*_simulationBox, *_data),
        exc::UserInputException,
        "The relaxation time of the Berendsen thermostat must not be shorter "
        "than the time step."
    );
}

TEST_F(TestThermostat, berendsenRejectsRelaxationTimeJustBelowTimestep)
{
    _thermostat =
        std::make_unique<thermostat::BerendsenThermostat>(300.0, 0.09);
    settings::TimingsSettings::setTimeStep(0.1);

    EXPECT_THROW(
        _thermostat->applyThermostat(*_simulationBox, *_data),
        exc::UserInputException
    );
}

TEST_F(TestThermostat, berendsenRelaxationTimeEqualToTimestepStaysFinite)
{
    _data->calculateTemperature(*_simulationBox);
    _thermostat = std::make_unique<thermostat::BerendsenThermostat>(0.0, 0.1);
    settings::TimingsSettings::setTimeStep(0.1);

    _thermostat->applyThermostat(*_simulationBox, *_data);

    EXPECT_TRUE(std::isfinite(_data->getTemperature()));
    EXPECT_TRUE(velocitiesAreFinite(*_simulationBox));
}

TEST_F(TestThermostat, berendsenNearZeroTemperatureStaysFinite)
{
    _thermostat =
        std::make_unique<thermostat::BerendsenThermostat>(300.0, 100.0);
    settings::TimingsSettings::setTimeStep(0.1);

    for (auto &atom : _simulationBox->getAtoms())
        atom->setVelocity({1e-12, 0.0, 0.0});

    _thermostat->applyThermostat(*_simulationBox, *_data);

    EXPECT_TRUE(std::isfinite(_data->getTemperature()));
    EXPECT_TRUE(velocitiesAreFinite(*_simulationBox));
}

TEST_F(TestThermostat, velocityRescalingRejectsZeroDegreesOfFreedom)
{
    _thermostat =
        std::make_unique<thermostat::VelocityRescalingThermostat>(300.0, 100.0);
    settings::TimingsSettings::setTimeStep(0.1);
    _simulationBox->setDegreesOfFreedom(0);

    EXPECT_THROW_MSG(
        _thermostat->applyThermostat(*_simulationBox, *_data),
        exc::UserInputException,
        "Cannot apply velocity rescaling to a system with zero degrees of "
        "freedom."
    );
}

TEST_F(TestThermostat, velocityRescalingRelaxationTimeShorterThanTimestep)
{
    _data->calculateTemperature(*_simulationBox);
    _thermostat = std::make_unique<thermostat::VelocityRescalingThermostat>(
        0.5 * _data->getTemperature(),
        0.01
    );
    settings::TimingsSettings::setTimeStep(0.1);

    _thermostat->applyThermostat(*_simulationBox, *_data);

    EXPECT_TRUE(std::isfinite(_data->getTemperature()));
    EXPECT_TRUE(velocitiesAreFinite(*_simulationBox));
}

namespace
{
    thermostat::NoseHooverThermostat noseHoover(double targetTemperature)
    {
        return thermostat::NoseHooverThermostat(
            targetTemperature,
            std::vector<double>{0.1, 0.2, 0.3},
            std::vector<double>{0.0, 0.0, 0.0},
            1.0
        );
    }

    constexpr auto NOSE_HOOVER_ZERO_TARGET =
        "Cannot apply the Nose-Hoover thermostat with a target temperature of "
        "zero or below.";
}   // namespace

TEST_F(TestThermostat, noseHooverRejectsZeroTargetTemperature)
{
    auto thermostat = noseHoover(0.0);
    settings::TimingsSettings::setTimeStep(0.1);

    EXPECT_THROW_MSG(
        thermostat.applyThermostat(*_simulationBox, *_data),
        exc::UserInputException,
        NOSE_HOOVER_ZERO_TARGET
    );
}

TEST_F(TestThermostat, noseHooverForcesRejectZeroTargetTemperature)
{
    auto thermostat = noseHoover(0.0);

    EXPECT_THROW_MSG(
        thermostat.applyThermostatOnForces(*_simulationBox),
        exc::UserInputException,
        NOSE_HOOVER_ZERO_TARGET
    );
}

TEST_F(TestThermostat, noseHooverRejectsZeroDegreesOfFreedom)
{
    auto thermostat = noseHoover(300.0);
    settings::TimingsSettings::setTimeStep(0.1);
    _simulationBox->setDegreesOfFreedom(0);

    EXPECT_THROW_MSG(
        thermostat.applyThermostat(*_simulationBox, *_data),
        exc::UserInputException,
        "Cannot apply the Nose-Hoover thermostat to a system with zero degrees "
        "of freedom."
    );
    EXPECT_THROW_MSG(
        thermostat.applyThermostatOnForces(*_simulationBox),
        exc::UserInputException,
        "Cannot apply the Nose-Hoover thermostat to a system with zero degrees "
        "of freedom."
    );
}

TEST_F(TestThermostat, noseHooverTinyPositiveTargetTemperatureStaysFinite)
{
    auto thermostat = noseHoover(1e-6);
    settings::TimingsSettings::setTimeStep(0.1);

    thermostat.applyThermostat(*_simulationBox, *_data);

    EXPECT_TRUE(std::isfinite(_data->getNoseHooverMomentumEnergy()));
    EXPECT_TRUE(std::isfinite(_data->getNoseHooverFrictionEnergy()));
}

TEST_F(TestThermostat, langevinTinyTimestepSigmaStaysFinite)
{
    const auto previousTimeStep = settings::TimingsSettings::getTimeStep();
    settings::TimingsSettings::setTimeStep(1e-12);
    const auto thermostat = thermostat::LangevinThermostat(300.0, 0.01);
    settings::TimingsSettings::setTimeStep(previousTimeStep);

    EXPECT_TRUE(std::isfinite(thermostat.getSigma()));
    EXPECT_GT(thermostat.getSigma(), 0.0);
}

/* ---------- LangevinThermostat ---------- */

TEST_F(TestThermostat, langevinConstructorComputesSigma)
{
    // sigma > 0 once friction and targetTemp are non-zero.
    const auto langevin = thermostat::LangevinThermostat(300.0, 0.1);
    EXPECT_GT(langevin.getSigma(), 0.0);
    EXPECT_DOUBLE_EQ(langevin.getFriction(), 0.1);
}

TEST_F(TestThermostat, langevinZeroFrictionHasZeroSigma)
{
    const auto langevin = thermostat::LangevinThermostat(300.0, 0.0);

    EXPECT_DOUBLE_EQ(langevin.getSigma(), 0.0);
    EXPECT_DOUBLE_EQ(langevin.getFriction(), 0.0);
}

TEST_F(TestThermostat, langevinSettersAndGetters)
{
    auto langevin = thermostat::LangevinThermostat(300.0, 0.1);

    langevin.setFriction(0.5);
    EXPECT_DOUBLE_EQ(langevin.getFriction(), 0.5);

    langevin.setSigma(2.0);
    EXPECT_DOUBLE_EQ(langevin.getSigma(), 2.0);
}

TEST_F(TestThermostat, langevinSetTargetTemperatureRecomputesSigma)
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

TEST_F(TestThermostat, langevinSetFrictionRecomputesSigma)
{
    auto langevin = thermostat::LangevinThermostat(300.0, 0.1);
    settings::TimingsSettings::setTimeStep(0.1);
    const auto sigmaAtFrictionPointOne = langevin.getSigma();

    langevin.setFriction(0.5);
    const auto sigmaAtFrictionPointFive = langevin.getSigma();

    EXPECT_NE(sigmaAtFrictionPointOne, sigmaAtFrictionPointFive);
    EXPECT_GT(sigmaAtFrictionPointFive, sigmaAtFrictionPointOne);
}

TEST_F(TestThermostat, langevinThermostatType)
{
    auto langevin = thermostat::LangevinThermostat(300.0, 0.1);
    EXPECT_EQ(langevin.getThermostatType(), ThermostatType::LANGEVIN);
}

/* ---------- NoseHooverThermostat ---------- */

TEST_F(TestThermostat, noseHooverThermostatType)
{
    auto thermostat = thermostat::NoseHooverThermostat(
        300.0,
        std::vector<double>{0.0, 0.0, 0.0},
        std::vector<double>{0.0, 0.0, 0.0},
        1.0e13
    );
    EXPECT_EQ(thermostat.getThermostatType(), ThermostatType::NOSE_HOOVER);
}

TEST_F(TestThermostat, noseHooverCouplingFrequencySetterGetter)
{
    auto thermostat = thermostat::NoseHooverThermostat(
        300.0,
        std::vector<double>{0.0, 0.0, 0.0},
        std::vector<double>{0.0, 0.0, 0.0},
        1.0e13
    );
    EXPECT_DOUBLE_EQ(thermostat.getCouplingFrequency(), 1.0e13);

    thermostat.setCouplingFrequency(5.0e12);
    EXPECT_DOUBLE_EQ(thermostat.getCouplingFrequency(), 5.0e12);
}

TEST_F(TestThermostat, noseHooverSetChiAtIndex)
{
    auto thermostat = thermostat::NoseHooverThermostat(
        300.0,
        std::vector<double>{0.0, 0.0, 0.0},
        std::vector<double>{0.0, 0.0, 0.0},
        1.0e13
    );
    thermostat.setChi(2U, 7.0);
    EXPECT_DOUBLE_EQ(thermostat.getChi()[2], 7.0);

    thermostat.setZeta(1U, 3.0);
    EXPECT_DOUBLE_EQ(thermostat.getZeta()[1], 3.0);
}

TEST_F(TestThermostat, noseHooverAppliesFiniteForceAndStateUpdates)
{
    auto thermostat = thermostat::NoseHooverThermostat(
        300.0,
        std::vector<double>{0.1, 0.2, 0.3},
        std::vector<double>{0.0, 0.0, 0.0},
        1.0
    );
    settings::TimingsSettings::setTimeStep(0.1);

    thermostat.applyThermostatOnForces(*_simulationBox);
    for (const auto &atom : _simulationBox->getAtoms())
        for (size_t axis = 0; axis < 3; ++axis)
            EXPECT_TRUE(std::isfinite(atom->getForce()[axis]));

    const auto chiBefore  = thermostat.getChi();
    const auto zetaBefore = thermostat.getZeta();
    thermostat.applyThermostat(*_simulationBox, *_data);

    EXPECT_TRUE(std::isfinite(_data->getTemperature()));
    EXPECT_TRUE(std::isfinite(_data->getNoseHooverMomentumEnergy()));
    EXPECT_TRUE(std::isfinite(_data->getNoseHooverFrictionEnergy()));
    EXPECT_NE(thermostat.getChi(), chiBefore);
    EXPECT_NE(thermostat.getZeta(), zetaBefore);
}
