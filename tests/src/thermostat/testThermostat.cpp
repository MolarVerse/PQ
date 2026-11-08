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

#include <cmath>
#include <memory>
#include <vector>

#include "berendsenThermostat.hpp"
#include "constants/internalConversionFactors.hpp"
#include "exceptions.hpp"
#include "langevinThermostat.hpp"
#include "noseHooverThermostat.hpp"
#include "physicalData.hpp"
#include "simulationBox.hpp"
#include "thermostatSettings.hpp"
#include "throwWithMessage.hpp"
#include "timingsSettings.hpp"
#include "units.hpp"
#include "velocityRescalingThermostat.hpp"

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
        _data->getTemperature().quantity_from_zero().numerical_value_in(K),
        sum(kineticEnergyAtomicVector) * TEMPERATURE_FACTOR /
            static_cast<double>(nDOF)
    );
}

TEST_F(TestThermostat, applyTemperatureRamping)
{
    _thermostat->setTemperatureIncrease(mstd::units::delta<K>(0.0));
    _thermostat->setTemperatureRampingSteps(0);
    _thermostat->setTemperatureRampingFrequency(1);
    _thermostat->setTargetTemperature(mstd::units::point<K>(300.0));

    _thermostat->applyTemperatureRamping();
    EXPECT_EQ(
        _thermostat->getTargetTemperature(),
        mstd::units::point<K>(300.0)
    );

    _thermostat->setTemperatureIncrease(mstd::units::delta<K>(1.0));
    _thermostat->setTemperatureRampingSteps(1);
    _thermostat->setTemperatureRampingFrequency(1);

    _thermostat->applyTemperatureRamping();
    EXPECT_EQ(
        _thermostat->getTargetTemperature(),
        mstd::units::point<K>(301.0)
    );

    _thermostat->applyTemperatureRamping();
    EXPECT_EQ(
        _thermostat->getTargetTemperature(),
        mstd::units::point<K>(301.0)
    );

    _thermostat->setTemperatureIncrease(mstd::units::delta<K>(1.0));
    _thermostat->setTemperatureRampingSteps(2);
    _thermostat->setTemperatureRampingFrequency(1);
    _thermostat->setTargetTemperature(mstd::units::point<K>(300.0));

    _thermostat->applyTemperatureRamping();
    EXPECT_EQ(
        _thermostat->getTargetTemperature(),
        mstd::units::point<K>(301.0)
    );

    _thermostat->applyTemperatureRamping();
    EXPECT_EQ(
        _thermostat->getTargetTemperature(),
        mstd::units::point<K>(302.0)
    );

    _thermostat->applyTemperatureRamping();
    EXPECT_EQ(
        _thermostat->getTargetTemperature(),
        mstd::units::point<K>(302.0)
    );

    _thermostat->setTemperatureIncrease(mstd::units::delta<K>(1.0));
    _thermostat->setTemperatureRampingSteps(4);
    _thermostat->setTemperatureRampingFrequency(2);
    _thermostat->setTargetTemperature(mstd::units::point<K>(300.0));

    _thermostat->applyTemperatureRamping();
    EXPECT_EQ(
        _thermostat->getTargetTemperature(),
        mstd::units::point<K>(300.0)
    );

    _thermostat->applyTemperatureRamping();
    EXPECT_EQ(
        _thermostat->getTargetTemperature(),
        mstd::units::point<K>(301.0)
    );

    _thermostat->applyTemperatureRamping();
    EXPECT_EQ(
        _thermostat->getTargetTemperature(),
        mstd::units::point<K>(301.0)
    );

    _thermostat->applyTemperatureRamping();
    EXPECT_EQ(
        _thermostat->getTargetTemperature(),
        mstd::units::point<K>(302.0)
    );

    _thermostat->applyTemperatureRamping();
    EXPECT_EQ(
        _thermostat->getTargetTemperature(),
        mstd::units::point<K>(302.0)
    );

    _thermostat->applyTemperatureRamping();
    EXPECT_EQ(
        _thermostat->getTargetTemperature(),
        mstd::units::point<K>(302.0)
    );
}

TEST_F(TestThermostat, applyThermostatBerendsen)
{
    _thermostat = std::make_unique<thermostat::BerendsenThermostat>(
        mstd::units::point<K>(300.0),
        100.0 * fs
    );
    settings::TimingsSettings::setTimeStep(0.1 * fs);

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
        _data->getTemperature().quantity_from_zero().numerical_value_in(K),
        oldTemperature * berendsenFactor * berendsenFactor
    );
}

/* ---------- VelocityRescalingThermostat ---------- */

TEST_F(TestThermostat, velocityRescalingTauSetterGetter)
{
    auto thermostat = thermostat::VelocityRescalingThermostat(
        mstd::units::point<K>(300.0),
        100.0 * fs
    );
    EXPECT_EQ(thermostat.getTau(), 100.0 * fs);

    thermostat.setTau(50.0 * fs);
    EXPECT_EQ(thermostat.getTau(), 50.0 * fs);
}

TEST_F(TestThermostat, velocityRescalingThermostatType)
{
    auto thermostat = thermostat::VelocityRescalingThermostat(
        mstd::units::point<K>(300.0),
        100.0 * fs
    );
    EXPECT_EQ(
        thermostat.getThermostatType(),
        ThermostatType::VELOCITY_RESCALING
    );
}

TEST_F(TestThermostat, velocityRescalingApplyDoesNotNaN)
{
    _thermostat = std::make_unique<thermostat::VelocityRescalingThermostat>(
        mstd::units::point<K>(300.0),
        100.0 * fs
    );
    settings::TimingsSettings::setTimeStep(0.1 * fs);

    _thermostat->applyThermostat(*_simulationBox, *_data);

    EXPECT_FALSE(
        std::isnan(
            _data->getTemperature().quantity_from_zero().numerical_value_in(K)
        )
    );
    EXPECT_FALSE(
        std::isinf(
            _data->getTemperature().quantity_from_zero().numerical_value_in(K)
        )
    );
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
    _thermostat = std::make_unique<thermostat::BerendsenThermostat>(
        mstd::units::point<K>(0.0),
        100.0 * fs
    );
    settings::TimingsSettings::setTimeStep(0.1 * fs);

    for (auto &atom : _simulationBox->getAtoms())
        atom->setVelocity({0.0, 0.0, 0.0});

    _thermostat->applyThermostat(*_simulationBox, *_data);

    EXPECT_TRUE(
        std::isfinite(
            _data->getTemperature().quantity_from_zero().numerical_value_in(K)
        )
    );
    for (const auto &atom : _simulationBox->getAtoms())
        for (size_t dimension = 0; dimension < 3; ++dimension)
            EXPECT_TRUE(std::isfinite(atom->getVelocity()[dimension]));
}

TEST_F(TestThermostat, berendsenRejectsPositiveTargetFromZero)
{
    _thermostat = std::make_unique<thermostat::BerendsenThermostat>(
        mstd::units::point<K>(300.0),
        100.0 * fs
    );
    settings::TimingsSettings::setTimeStep(0.1 * fs);

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
    _thermostat = std::make_unique<thermostat::VelocityRescalingThermostat>(
        mstd::units::point<K>(0.0),
        100.0 * fs
    );
    settings::TimingsSettings::setTimeStep(0.1 * fs);

    for (auto &atom : _simulationBox->getAtoms())
        atom->setVelocity({0.0, 0.0, 0.0});

    _thermostat->applyThermostat(*_simulationBox, *_data);

    EXPECT_TRUE(
        std::isfinite(
            _data->getTemperature().quantity_from_zero().numerical_value_in(K)
        )
    );
    for (const auto &atom : _simulationBox->getAtoms())
        for (size_t dimension = 0; dimension < 3; ++dimension)
            EXPECT_TRUE(std::isfinite(atom->getVelocity()[dimension]));
}

TEST_F(TestThermostat, velocityRescalingRejectsPositiveTargetFromZero)
{
    _thermostat = std::make_unique<thermostat::VelocityRescalingThermostat>(
        mstd::units::point<K>(300.0),
        100.0 * fs
    );
    settings::TimingsSettings::setTimeStep(0.1 * fs);

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
    bool velocitiesAreFinite(const molsys::SimulationBox &simulationBox)
    {
        for (const auto &atom : simulationBox.getAtoms())
        {
            for (size_t axis = 0; axis < 3; ++axis)
            {
                if (!std::isfinite(atom->getVelocity()[axis]))
                {
                    return false;
                }
            }
        }
        return true;
    }
}   // namespace

TEST_F(TestThermostat, berendsenRelaxationTimeEqualToTimestepStaysFinite)
{
    _data->calculateTemperature(*_simulationBox);
    _thermostat = std::make_unique<thermostat::BerendsenThermostat>(
        mstd::units::point<K>(0.0),
        0.1 * fs
    );
    settings::TimingsSettings::setTimeStep(0.1 * fs);

    _thermostat->applyThermostat(*_simulationBox, *_data);

    EXPECT_TRUE(
        std::isfinite(
            _data->getTemperature().quantity_from_zero().numerical_value_in(K)
        )
    );
    EXPECT_TRUE(velocitiesAreFinite(*_simulationBox));
}

TEST_F(TestThermostat, berendsenNearZeroTemperatureStaysFinite)
{
    _thermostat = std::make_unique<thermostat::BerendsenThermostat>(
        mstd::units::point<K>(300.0),
        100.0 * fs
    );
    settings::TimingsSettings::setTimeStep(0.1 * fs);

    for (auto &atom : _simulationBox->getAtoms())
        atom->setVelocity({1e-12, 0.0, 0.0});

    _thermostat->applyThermostat(*_simulationBox, *_data);

    EXPECT_TRUE(
        std::isfinite(
            _data->getTemperature().quantity_from_zero().numerical_value_in(K)
        )
    );
    EXPECT_TRUE(velocitiesAreFinite(*_simulationBox));
}

TEST_F(TestThermostat, velocityRescalingRejectsZeroDegreesOfFreedom)
{
    _thermostat = std::make_unique<thermostat::VelocityRescalingThermostat>(
        mstd::units::point<K>(300.0),
        100.0 * fs
    );
    settings::TimingsSettings::setTimeStep(0.1 * fs);
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
        Temperature{0.5 * _data->getTemperature().quantity_from_zero()},
        0.01 * fs
    );
    settings::TimingsSettings::setTimeStep(0.1 * fs);

    _thermostat->applyThermostat(*_simulationBox, *_data);

    EXPECT_TRUE(
        std::isfinite(
            _data->getTemperature().quantity_from_zero().numerical_value_in(K)
        )
    );
    EXPECT_TRUE(velocitiesAreFinite(*_simulationBox));
}

namespace
{
    thermostat::NoseHooverThermostat noseHoover(double targetTemperature)
    {
        return thermostat::NoseHooverThermostat(
            mstd::units::point<K>(targetTemperature),
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
    settings::TimingsSettings::setTimeStep(0.1 * fs);

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
    settings::TimingsSettings::setTimeStep(0.1 * fs);
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
    settings::TimingsSettings::setTimeStep(0.1 * fs);

    thermostat.applyThermostat(*_simulationBox, *_data);

    EXPECT_TRUE(std::isfinite(_data->getNoseHooverMomentumEnergy()));
    EXPECT_TRUE(std::isfinite(_data->getNoseHooverFrictionEnergy()));
}

TEST_F(TestThermostat, langevinTinyTimestepSigmaStaysFinite)
{
    const auto previousTimeStep = settings::TimingsSettings::getTimeStep();
    settings::TimingsSettings::setTimeStep(1e-12 * fs);
    const auto thermostat =
        thermostat::LangevinThermostat(mstd::units::point<K>(300.0), 0.01);
    settings::TimingsSettings::setTimeStep(previousTimeStep);

    EXPECT_TRUE(std::isfinite(thermostat.getSigma()));
    EXPECT_GT(thermostat.getSigma(), 0.0);
}

/* ---------- LangevinThermostat ---------- */

TEST_F(TestThermostat, langevinConstructorComputesSigma)
{
    // sigma > 0 once friction and targetTemp are non-zero.
    const auto langevin =
        thermostat::LangevinThermostat(mstd::units::point<K>(300.0), 0.1);
    EXPECT_GT(langevin.getSigma(), 0.0);
    EXPECT_DOUBLE_EQ(langevin.getFriction(), 0.1);
}

TEST_F(TestThermostat, langevinZeroFrictionHasZeroSigma)
{
    const auto langevin =
        thermostat::LangevinThermostat(mstd::units::point<K>(300.0), 0.0);

    EXPECT_DOUBLE_EQ(langevin.getSigma(), 0.0);
    EXPECT_DOUBLE_EQ(langevin.getFriction(), 0.0);
}

TEST_F(TestThermostat, langevinSettersAndGetters)
{
    auto langevin =
        thermostat::LangevinThermostat(mstd::units::point<K>(300.0), 0.1);

    langevin.setFriction(0.5);
    EXPECT_DOUBLE_EQ(langevin.getFriction(), 0.5);

    langevin.setSigma(2.0);
    EXPECT_DOUBLE_EQ(langevin.getSigma(), 2.0);
}

TEST_F(TestThermostat, langevinSetTargetTemperatureRecomputesSigma)
{
    auto langevin =
        thermostat::LangevinThermostat(mstd::units::point<K>(300.0), 0.1);
    settings::TimingsSettings::setTimeStep(0.1 * fs);
    const auto sigmaAt300 = langevin.getSigma();

    langevin.setTargetTemperature(mstd::units::point<K>(600.0));
    const auto sigmaAt600 = langevin.getSigma();

    // Setting a higher target temperature must update sigma (the
    // Langevin Gaussian-noise amplitude scales with sqrt(kBT)).
    EXPECT_NE(sigmaAt300, sigmaAt600);
    EXPECT_GT(sigmaAt600, sigmaAt300);
}

TEST_F(TestThermostat, langevinSetFrictionRecomputesSigma)
{
    auto langevin =
        thermostat::LangevinThermostat(mstd::units::point<K>(300.0), 0.1);
    settings::TimingsSettings::setTimeStep(0.1 * fs);
    const auto sigmaAtFrictionPointOne = langevin.getSigma();

    langevin.setFriction(0.5);
    const auto sigmaAtFrictionPointFive = langevin.getSigma();

    EXPECT_NE(sigmaAtFrictionPointOne, sigmaAtFrictionPointFive);
    EXPECT_GT(sigmaAtFrictionPointFive, sigmaAtFrictionPointOne);
}

TEST_F(TestThermostat, langevinThermostatType)
{
    auto langevin =
        thermostat::LangevinThermostat(mstd::units::point<K>(300.0), 0.1);
    EXPECT_EQ(langevin.getThermostatType(), ThermostatType::LANGEVIN);
}

/* ---------- NoseHooverThermostat ---------- */

TEST_F(TestThermostat, noseHooverThermostatType)
{
    auto thermostat = thermostat::NoseHooverThermostat(
        mstd::units::point<K>(300.0),
        std::vector<double>{0.0, 0.0, 0.0},
        std::vector<double>{0.0, 0.0, 0.0},
        1.0e13
    );
    EXPECT_EQ(thermostat.getThermostatType(), ThermostatType::NOSE_HOOVER);
}

TEST_F(TestThermostat, noseHooverCouplingFrequencySetterGetter)
{
    auto thermostat = thermostat::NoseHooverThermostat(
        mstd::units::point<K>(300.0),
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
        mstd::units::point<K>(300.0),
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
        mstd::units::point<K>(300.0),
        std::vector<double>{0.1, 0.2, 0.3},
        std::vector<double>{0.0, 0.0, 0.0},
        1.0
    );
    settings::TimingsSettings::setTimeStep(0.1 * fs);

    thermostat.applyThermostatOnForces(*_simulationBox);
    for (const auto &atom : _simulationBox->getAtoms())
        for (size_t axis = 0; axis < 3; ++axis)
            EXPECT_TRUE(std::isfinite(atom->getForce()[axis]));

    const auto chiBefore  = thermostat.getChi();
    const auto zetaBefore = thermostat.getZeta();
    thermostat.applyThermostat(*_simulationBox, *_data);

    EXPECT_TRUE(
        std::isfinite(
            _data->getTemperature().quantity_from_zero().numerical_value_in(K)
        )
    );
    EXPECT_TRUE(std::isfinite(_data->getNoseHooverMomentumEnergy()));
    EXPECT_TRUE(std::isfinite(_data->getNoseHooverFrictionEnergy()));
    EXPECT_NE(thermostat.getChi(), chiBefore);
    EXPECT_NE(thermostat.getZeta(), zetaBefore);
}

/* ---------- additional value-level coverage ---------- */

TEST_F(TestThermostat, temperatureSetterGetter)
{
    _thermostat->setTemperature(mstd::units::point<K>(123.0));
    EXPECT_EQ(_thermostat->getTemperature(), mstd::units::point<K>(123.0));

    _thermostat->setTargetTemperature(mstd::units::point<K>(456.0));
    EXPECT_EQ(
        _thermostat->getTargetTemperature(),
        mstd::units::point<K>(456.0)
    );

    _thermostat->setTemperatureIncrease(mstd::units::delta<K>(7.5));
    EXPECT_EQ(
        _thermostat->getTemperatureIncrease(),
        mstd::units::delta<K>(7.5)
    );
}

TEST_F(TestThermostat, temperatureRampingUpdatesActualTargetTemperature)
{
    settings::ThermostatSettings::setActualTargetTemperature(300.0);

    _thermostat->setTargetTemperature(mstd::units::point<K>(300.0));
    _thermostat->setTemperatureIncrease(mstd::units::delta<K>(2.5));
    _thermostat->setTemperatureRampingSteps(2);
    _thermostat->setTemperatureRampingFrequency(1);

    _thermostat->applyTemperatureRamping();
    EXPECT_DOUBLE_EQ(
        settings::ThermostatSettings::getActualTargetTemperature(),
        302.5
    );

    _thermostat->applyTemperatureRamping();
    EXPECT_DOUBLE_EQ(
        settings::ThermostatSettings::getActualTargetTemperature(),
        305.0
    );

    settings::ThermostatSettings::setActualTargetTemperature(0.0);
}

TEST_F(TestThermostat, berendsenStoresMeasuredTemperature)
{
    _thermostat = std::make_unique<thermostat::BerendsenThermostat>(
        mstd::units::point<K>(300.0),
        100.0 * fs
    );
    settings::TimingsSettings::setTimeStep(0.1 * fs);

    _data->calculateTemperature(*_simulationBox);
    const auto oldTemperature = _data->getTemperature();

    _thermostat->applyThermostat(*_simulationBox, *_data);

    EXPECT_EQ(_thermostat->getTemperature(), oldTemperature);
}

TEST_F(TestThermostat, berendsenScalesVelocitiesBySqrtOfTemperatureRatio)
{
    _thermostat = std::make_unique<thermostat::BerendsenThermostat>(
        mstd::units::point<K>(300.0),
        100.0 * fs
    );
    settings::TimingsSettings::setTimeStep(0.1 * fs);

    _data->calculateTemperature(*_simulationBox);
    const auto oldTemperature = _data->getTemperature();
    const auto oldVelocity    = _simulationBox->getAtoms()[1]->getVelocity();

    _thermostat->applyThermostat(*_simulationBox, *_data);

    const auto factor =
        std::sqrt((_data->getTemperature().quantity_from_zero() /
                   oldTemperature.quantity_from_zero())
                      .numerical_value_in(one));
    const auto newVelocity = _simulationBox->getAtoms()[1]->getVelocity();
    for (size_t axis = 0; axis < 3; ++axis)
        EXPECT_NEAR(newVelocity[axis], factor * oldVelocity[axis], 1e-12);
}

namespace
{
    std::vector<linalg::Vec3D> saveVelocities(
        const molsys::SimulationBox &simulationBox
    )
    {
        std::vector<linalg::Vec3D> velocities;
        for (const auto &atom : simulationBox.getAtoms())
            velocities.push_back(atom->getVelocity());
        return velocities;
    }

    void restoreVelocities(
        molsys::SimulationBox            &simulationBox,
        const std::vector<linalg::Vec3D> &velocities
    )
    {
        for (size_t i = 0; i < velocities.size(); ++i)
            simulationBox.getAtoms()[i]->setVelocity(velocities[i]);
    }
}   // namespace

/**
 * lambda = 1 + timeRatio * (Ttarget/T - 1) + 2 sqrt(timeRatio Ttarget/T / dof)
 * * N(0,1). Repeated draws from identical states must reproduce the mean
 * and the standard deviation, and the velocities must be scaled by
 * sqrt(lambda) consistent with the stored temperature.
 */
TEST_F(TestThermostat, velocityRescalingLambdaStatistics)
{
    settings::TimingsSettings::setTimeStep(0.1 * fs);

    _data->calculateTemperature(*_simulationBox);
    const auto temperature = _data->getTemperature();
    const auto initial     = saveVelocities(*_simulationBox);
    const auto dof = static_cast<double>(_simulationBox->getDegreesOfFreedom());

    const auto tempRatio = 10.0;
    const auto timeRatio = 0.1 / 10.0;

    auto thermostat = thermostat::VelocityRescalingThermostat(
        Temperature{tempRatio * temperature.quantity_from_zero()},
        10.0 * fs
    );

    const auto expectedMean = 1.0 + (timeRatio * (tempRatio - 1.0));
    const auto expectedStd  = 2.0 * std::sqrt(timeRatio * tempRatio / dof);

    constexpr size_t nSamples = 4000;
    double           sum      = 0.0;
    double           sumSq    = 0.0;

    for (size_t i = 0; i < nSamples; ++i)
    {
        restoreVelocities(*_simulationBox, initial);
        thermostat.applyThermostat(*_simulationBox, *_data);

        const auto lambda = (_data->getTemperature().quantity_from_zero() /
                             temperature.quantity_from_zero())
                                .numerical_value_in(one);
        sum   += lambda;
        sumSq += lambda * lambda;

        const auto scaled = _simulationBox->getAtoms()[1]->getVelocity();
        for (size_t axis = 0; axis < 3; ++axis)
        {
            ASSERT_NEAR(
                scaled[axis],
                std::sqrt(lambda) * initial[1][axis],
                1e-10
            );
        }
    }

    const auto mean = sum / nSamples;
    const auto var  = (sumSq / nSamples) - (mean * mean);

    EXPECT_NEAR(mean, expectedMean, 5.0 * expectedStd / std::sqrt(nSamples));
    EXPECT_NEAR(std::sqrt(var), expectedStd, 0.1 * expectedStd);
    EXPECT_EQ(thermostat.getTemperature(), temperature);
}

TEST_F(TestThermostat, velocityRescalingNoStochasticTermAtEquilibriumLimit)
{
    // timestep << tau: lambda -> 1, temperature must stay (nearly) unchanged
    settings::TimingsSettings::setTimeStep(1e-9 * fs);

    _data->calculateTemperature(*_simulationBox);
    const auto temperature = _data->getTemperature();

    auto thermostat = thermostat::VelocityRescalingThermostat(
        Temperature{3.0 * temperature.quantity_from_zero()},
        100.0 * fs
    );
    thermostat.applyThermostat(*_simulationBox, *_data);

    EXPECT_NEAR(
        _data->getTemperature().quantity_from_zero().numerical_value_in(K),
        temperature.quantity_from_zero().numerical_value_in(K),
        1e-3 * temperature.quantity_from_zero().numerical_value_in(K)
    );
}

/**
 * Independent re-implementation of one Nose-Hoover step in SI-time units.
 */
TEST_F(TestThermostat, noseHooverMatchesReferenceStep)
{
    settings::TimingsSettings::setTimeStep(0.1 * fs);

    const auto targetTemp = 300.0;
    const auto freq       = 1.0;
    auto       chi        = std::vector<double>{0.1, 0.2, 0.3};
    auto       zeta       = std::vector<double>{0.0, 0.0, 0.0};

    auto thermostat = thermostat::NoseHooverThermostat(
        mstd::units::point<K>(targetTemp),
        chi,
        zeta,
        freq
    );

    thermostat.applyThermostat(*_simulationBox, *_data);

    const auto temperature = _data->getTemperature();
    const auto dof = static_cast<double>(_simulationBox->getDegreesOfFreedom());
    const auto timeStep          = 0.1e-15;
    const auto freq2             = freq * freq;
    const auto boltzmannConstant = BOLTZMANN_CONSTANT_IN_KCAL_PER_MOL;
    const auto kineticEnergy =
        boltzmannConstant *
        temperature.quantity_from_zero().numerical_value_in(K);
    const auto eTar = boltzmannConstant * targetTemp;

    auto chiTemp =
        ((kineticEnergy - eTar) * dof) - (chi[0] * chi[1] / eTar * freq2);
    chi[0]      += timeStep * chiTemp;
    auto helper  = chi[0] / (eTar * dof) * freq2;
    zeta[0]     += helper * timeStep;
    auto ratio   = helper * chi[0];
    auto eMom    = ratio;
    auto eFri    = dof * zeta[0];
    for (size_t i = 1; i < chi.size() - 1; ++i)
    {
        chiTemp  = ratio - eTar - chi[i] * chi[i + 1] / eTar * freq2;
        chi[i]  += timeStep * chiTemp;
        helper   = chi[i] / eTar * freq2;
        zeta[i] += helper * timeStep;
        ratio    = helper * chi[i];
        eMom    += ratio;
        eFri    += zeta[i];
    }

    for (size_t i = 0; i < chi.size(); ++i)
    {
        EXPECT_DOUBLE_EQ(thermostat.getChi()[i], chi[i]);
        EXPECT_DOUBLE_EQ(thermostat.getZeta()[i], zeta[i]);
    }
    EXPECT_DOUBLE_EQ(_data->getNoseHooverMomentumEnergy(), eMom);
    EXPECT_DOUBLE_EQ(_data->getNoseHooverFrictionEnergy(), eFri);
    EXPECT_EQ(thermostat.getTemperature(), temperature);
}

TEST_F(TestThermostat, noseHooverForceMatchesReference)
{
    const auto targetTemp = 250.0;
    const auto freq       = 2.0;
    const auto chi0       = 0.4;

    auto thermostat = thermostat::NoseHooverThermostat(
        mstd::units::point<K>(targetTemp),
        std::vector<double>{chi0, 0.2, 0.3},
        std::vector<double>{0.0, 0.0, 0.0},
        freq
    );

    for (auto &atom : _simulationBox->getAtoms())
        atom->setForce(linalg::Vec3D(0.0, 0.0, 0.0));

    thermostat.applyThermostatOnForces(*_simulationBox);

    const auto dof = static_cast<double>(_simulationBox->getDegreesOfFreedom());
    const auto factor =
        chi0 * freq * freq /
        (BOLTZMANN_CONSTANT_IN_KCAL_PER_MOL * targetTemp * dof) *
        MOMENTUM_TO_FORCE;

    for (const auto &atom : _simulationBox->getAtoms())
    {
        for (size_t axis = 0; axis < 3; ++axis)
        {
            EXPECT_DOUBLE_EQ(
                atom->getForce()[axis],
                -factor * atom->getVelocity()[axis] * atom->getMass()
            );
        }
    }
}

TEST_F(TestThermostat, noseHooverRejectsNegativeTargetTemperature)
{
    auto thermostat = noseHoover(-5.0);
    settings::TimingsSettings::setTimeStep(0.1 * fs);

    EXPECT_THROW_MSG(
        thermostat.applyThermostat(*_simulationBox, *_data),
        exc::UserInputException,
        NOSE_HOOVER_ZERO_TARGET
    );
}

TEST_F(TestThermostat, langevinSigmaMatchesReference)
{
    settings::TimingsSettings::setTimeStep(0.1 * fs);

    const auto friction = 0.1;
    const auto langevin =
        thermostat::LangevinThermostat(mstd::units::point<K>(300.0), friction);

    const auto unitConversion = M2_TO_ANGSTROM2 * KG_TO_GRAM / FS_TO_S;
    const auto expected       = std::sqrt(
        4.0 * friction * UNIVERSAL_GAS_CONSTANT * unitConversion * 300.0 / 0.1
    );

    EXPECT_DOUBLE_EQ(langevin.getSigma(), expected);

    // sigma scales with sqrt(T)
    auto langevin4 =
        thermostat::LangevinThermostat(mstd::units::point<K>(1200.0), friction);
    EXPECT_NEAR(
        langevin4.getSigma(),
        2.0 * langevin.getSigma(),
        1e-12 * expected
    );
}
