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
#include "exceptions.hpp"
#include "generalSettings.hpp"
#include "langevinThermostat.hpp"
#include "noseHooverThermostat.hpp"
#include "physicalData.hpp"
#include "simulationBox.hpp"
#include "thermostatSettings.hpp"
#include "throwWithMessage.hpp"
#include "unit_types.hpp"
#include "units.hpp"
#include "velocityRescalingThermostat.hpp"

namespace
{
    // K per (g/mol * A^2/s^2)
    const double temperatureFactor = ((1.0 * g_per_mol) * (1.0 * angstrom / s) *
                                      (1.0 * angstrom / s) / gas_constant)
                                         .numerical_value_in(K);
}   // namespace

TEST_F(TestThermostat, calculateTemperature)
{
    _thermostat->applyThermostat(
        *_simulationBox,
        *_data,
        toTemperature(300.0),
        0.1 * fs
    );

    const auto mol1 = _simulationBox->getMolecule(0);
    const auto velocity_mol1_atom1 =
        mol1.getAtomVelocity(AtomIndex{0}).numerical_value_in(angstrom / s);
    const auto velocity_mol1_atom2 =
        mol1.getAtomVelocity(AtomIndex{1}).numerical_value_in(angstrom / s);
    const auto mass_mol1_atom1 =
        mol1.getAtomMass(AtomIndex{0}).numerical_value_in(g_per_mol);
    const auto mass_mol1_atom2 =
        mol1.getAtomMass(AtomIndex{1}).numerical_value_in(g_per_mol);

    const auto mol2 = _simulationBox->getMolecule(1);
    const auto velocity_mol2_atom1 =
        mol2.getAtomVelocity(AtomIndex{0}).numerical_value_in(angstrom / s);
    const auto mass_mol2_atom1 =
        mol2.getAtomMass(AtomIndex{0}).numerical_value_in(g_per_mol);

    const auto kineticEnergyAtomicVector =
        mass_mol1_atom1 * velocity_mol1_atom1 * velocity_mol1_atom1 +
        mass_mol1_atom2 * velocity_mol1_atom2 * velocity_mol1_atom2 +
        mass_mol2_atom1 * velocity_mol2_atom1 * velocity_mol2_atom1;

    const auto nDOF = _simulationBox->getDegreesOfFreedom();

    EXPECT_EQ(
        _data->getTemperature().quantity_from_zero().numerical_value_in(K),
        sum(kineticEnergyAtomicVector) * temperatureFactor /
            static_cast<double>(nDOF)
    );
}

TEST_F(TestThermostat, applyThermostatBerendsen)
{
    _thermostat = std::make_unique<thermostat::BerendsenThermostat>(100.0 * fs);

    const auto mol1 = _simulationBox->getMolecule(0);
    const auto velocity_mol1_atom1 =
        mol1.getAtomVelocity(AtomIndex{0}).numerical_value_in(angstrom / s);
    const auto velocity_mol1_atom2 =
        mol1.getAtomVelocity(AtomIndex{1}).numerical_value_in(angstrom / s);
    const auto mass_mol1_atom1 =
        mol1.getAtomMass(AtomIndex{0}).numerical_value_in(g_per_mol);
    const auto mass_mol1_atom2 =
        mol1.getAtomMass(AtomIndex{1}).numerical_value_in(g_per_mol);

    const auto mol2 = _simulationBox->getMolecule(1);
    const auto velocity_mol2_atom1 =
        mol2.getAtomVelocity(AtomIndex{0}).numerical_value_in(angstrom / s);
    const auto mass_mol2_atom1 =
        mol2.getAtomMass(AtomIndex{0}).numerical_value_in(g_per_mol);

    const auto kineticEnergyAtomicVector =
        mass_mol1_atom1 * velocity_mol1_atom1 * velocity_mol1_atom1 +
        mass_mol1_atom2 * velocity_mol1_atom2 * velocity_mol1_atom2 +
        mass_mol2_atom1 * velocity_mol2_atom1 * velocity_mol2_atom1;

    const auto nDOF = _simulationBox->getDegreesOfFreedom();

    const auto oldTemperature = sum(kineticEnergyAtomicVector) *
                                temperatureFactor / static_cast<double>(nDOF);

    const auto berendsenFactor =
        ::sqrt(1.0 + (0.1 / 100.0 * (300.0 / oldTemperature - 1.0)));

    _thermostat->applyThermostat(
        *_simulationBox,
        *_data,
        toTemperature(300.0),
        0.1 * fs
    );

    EXPECT_EQ(
        _data->getTemperature().quantity_from_zero().numerical_value_in(K),
        oldTemperature * berendsenFactor * berendsenFactor
    );
}

/* ---------- VelocityRescalingThermostat ---------- */

TEST_F(TestThermostat, velocityRescalingTauSetterGetter)
{
    auto thermostat = thermostat::VelocityRescalingThermostat(100.0 * fs);
    EXPECT_EQ(thermostat.getTau(), 100.0 * fs);

    thermostat.setTau(50.0 * fs);
    EXPECT_EQ(thermostat.getTau(), 50.0 * fs);
}

TEST_F(TestThermostat, velocityRescalingThermostatType)
{
    auto thermostat = thermostat::VelocityRescalingThermostat(100.0 * fs);
    EXPECT_EQ(
        thermostat.getThermostatType(),
        ThermostatType::VELOCITY_RESCALING
    );
}

TEST_F(TestThermostat, velocityRescalingApplyDoesNotNaN)
{
    _thermostat =
        std::make_unique<thermostat::VelocityRescalingThermostat>(100.0 * fs);

    _thermostat->applyThermostat(
        *_simulationBox,
        *_data,
        toTemperature(300.0),
        0.1 * fs
    );

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
            EXPECT_FALSE(isnan(atom->getVelocity()[i]));
            EXPECT_FALSE(isinf(atom->getVelocity()[i]));
        }
    }
}

TEST_F(TestThermostat, berendsenZeroTemperatureDoesNotNaN)
{
    _thermostat = std::make_unique<thermostat::BerendsenThermostat>(100.0 * fs);

    for (auto &atom : _simulationBox->getAtoms())
        atom->setVelocity(linalg::Vec3D{0.0, 0.0, 0.0} * angstrom / s);

    _thermostat->applyThermostat(
        *_simulationBox,
        *_data,
        toTemperature(0.0),
        0.1 * fs
    );

    EXPECT_TRUE(
        std::isfinite(
            _data->getTemperature().quantity_from_zero().numerical_value_in(K)
        )
    );
    for (const auto &atom : _simulationBox->getAtoms())
        for (size_t dimension = 0; dimension < 3; ++dimension)
            EXPECT_TRUE(isfinite(atom->getVelocity()[dimension]));
}

TEST_F(TestThermostat, berendsenRejectsPositiveTargetFromZero)
{
    _thermostat = std::make_unique<thermostat::BerendsenThermostat>(100.0 * fs);

    for (auto &atom : _simulationBox->getAtoms())
        atom->setVelocity(linalg::Vec3D{0.0, 0.0, 0.0} * angstrom / s);

    EXPECT_THROW_MSG(
        _thermostat->applyThermostat(
            *_simulationBox,
            *_data,
            toTemperature(300.0),
            0.1 * fs
        ),
        exc::UserInputException,
        "Cannot apply Berendsen coupling to a zero-temperature system with a "
        "positive target temperature. Initialize velocities first."
    );
}

TEST_F(TestThermostat, velocityRescalingZeroTemperatureDoesNotNaN)
{
    _thermostat =
        std::make_unique<thermostat::VelocityRescalingThermostat>(100.0 * fs);

    for (auto &atom : _simulationBox->getAtoms())
        atom->setVelocity(linalg::Vec3D{0.0, 0.0, 0.0} * angstrom / s);

    _thermostat->applyThermostat(
        *_simulationBox,
        *_data,
        toTemperature(0.0),
        0.1 * fs
    );

    EXPECT_TRUE(
        std::isfinite(
            _data->getTemperature().quantity_from_zero().numerical_value_in(K)
        )
    );
    for (const auto &atom : _simulationBox->getAtoms())
        for (size_t dimension = 0; dimension < 3; ++dimension)
            EXPECT_TRUE(isfinite(atom->getVelocity()[dimension]));
}

TEST_F(TestThermostat, velocityRescalingRejectsPositiveTargetFromZero)
{
    _thermostat =
        std::make_unique<thermostat::VelocityRescalingThermostat>(100.0 * fs);

    for (auto &atom : _simulationBox->getAtoms())
        atom->setVelocity(linalg::Vec3D{0.0, 0.0, 0.0} * angstrom / s);

    EXPECT_THROW_MSG(
        _thermostat->applyThermostat(
            *_simulationBox,
            *_data,
            toTemperature(300.0),
            0.1 * fs
        ),
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
                if (!isfinite(atom->getVelocity()[axis]))
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
    _thermostat = std::make_unique<thermostat::BerendsenThermostat>(0.1 * fs);

    _thermostat
        ->applyThermostat(*_simulationBox, *_data, toTemperature(0), 0.1 * fs);

    EXPECT_TRUE(
        std::isfinite(
            _data->getTemperature().quantity_from_zero().numerical_value_in(K)
        )
    );
    EXPECT_TRUE(velocitiesAreFinite(*_simulationBox));
}

TEST_F(TestThermostat, berendsenNearZeroTemperatureStaysFinite)
{
    _thermostat = std::make_unique<thermostat::BerendsenThermostat>(100.0 * fs);

    for (auto &atom : _simulationBox->getAtoms())
        atom->setVelocity(linalg::Vec3D{1e-12, 0.0, 0.0} * angstrom / s);

    _thermostat->applyThermostat(
        *_simulationBox,
        *_data,
        toTemperature(300.0),
        0.1 * fs
    );

    EXPECT_TRUE(
        std::isfinite(
            _data->getTemperature().quantity_from_zero().numerical_value_in(K)
        )
    );
    EXPECT_TRUE(velocitiesAreFinite(*_simulationBox));
}

TEST_F(TestThermostat, velocityRescalingRejectsZeroDegreesOfFreedom)
{
    _thermostat =
        std::make_unique<thermostat::VelocityRescalingThermostat>(100.0 * fs);
    _simulationBox->setDegreesOfFreedom(0);

    EXPECT_THROW_MSG(
        _thermostat->applyThermostat(
            *_simulationBox,
            *_data,
            toTemperature(300.0),
            0.1 * fs
        ),
        exc::UserInputException,
        "Cannot apply velocity rescaling to a system with zero degrees of "
        "freedom."
    );
}

TEST_F(TestThermostat, velocityRescalingRelaxationTimeShorterThanTimestep)
{
    _data->calculateTemperature(*_simulationBox);
    _thermostat =
        std::make_unique<thermostat::VelocityRescalingThermostat>(0.01 * fs);

    _thermostat->applyThermostat(
        *_simulationBox,
        *_data,
        Temperature{0.5 * _data->getTemperature().quantity_from_zero()},
        0.1 * fs
    );

    EXPECT_TRUE(
        std::isfinite(
            _data->getTemperature().quantity_from_zero().numerical_value_in(K)
        )
    );
    EXPECT_TRUE(velocitiesAreFinite(*_simulationBox));
}

namespace
{
    NoseHooverSettings noseHooverSettings()
    {
        NoseHooverSettings settings;
        settings.setNoseHooverCouplingFrequencyByUser(1.0 * Hz);
        const auto chis = std::map<size_t, Action>{
            {0, 0.1 * kcal_per_mol * s},
            {1, 0.2 * kcal_per_mol * s},
            {2, 0.3 * kcal_per_mol * s}
        };
        const auto zetas =
            std::map<size_t, Quantity>{{0, 0.0}, {1, 0.0}, {2, 0.0}};
        settings.setChiByUser(chis);
        settings.setZetaByUser(zetas);
        return settings;
    }

    // the thermostat only stores a reference, so the settings must outlive it
    thermostat::NoseHooverThermostat noseHoover(
        const NoseHooverSettings &settings
    )
    {
        return thermostat::NoseHooverThermostat(settings);
    }

    constexpr auto NOSE_HOOVER_ZERO_TARGET =
        "Cannot apply the Nose-Hoover thermostat with a target temperature of "
        "zero or below.";
}   // namespace

TEST_F(TestThermostat, noseHooverRejectsZeroTargetTemperature)
{
    const auto nhSettings = noseHooverSettings();
    auto       thermostat = noseHoover(nhSettings);

    EXPECT_THROW_MSG(
        thermostat.applyThermostat(
            *_simulationBox,
            *_data,
            toTemperature(0.0),
            0.1 * fs
        ),
        exc::UserInputException,
        NOSE_HOOVER_ZERO_TARGET
    );
}

TEST_F(TestThermostat, noseHooverForcesRejectZeroTargetTemperature)
{
    const auto nhSettings = noseHooverSettings();
    auto       thermostat = noseHoover(nhSettings);

    EXPECT_THROW_MSG(
        thermostat.applyThermostatOnForces(*_simulationBox, toTemperature(0.0)),
        exc::UserInputException,
        NOSE_HOOVER_ZERO_TARGET
    );
}

TEST_F(TestThermostat, noseHooverRejectsZeroDegreesOfFreedom)
{
    const auto nhSettings = noseHooverSettings();
    auto       thermostat = noseHoover(nhSettings);
    _simulationBox->setDegreesOfFreedom(0);

    EXPECT_THROW_MSG(
        thermostat.applyThermostat(
            *_simulationBox,
            *_data,
            toTemperature(300.0),
            0.1 * fs
        ),
        exc::UserInputException,
        "Cannot apply the Nose-Hoover thermostat to a system with zero degrees "
        "of freedom."
    );
    EXPECT_THROW_MSG(
        thermostat
            .applyThermostatOnForces(*_simulationBox, toTemperature(300.0)),
        exc::UserInputException,
        "Cannot apply the Nose-Hoover thermostat to a system with zero degrees "
        "of freedom."
    );
}

TEST_F(TestThermostat, noseHooverTinyPositiveTargetTemperatureStaysFinite)
{
    const auto nhSettings = noseHooverSettings();
    auto       thermostat = noseHoover(nhSettings);

    thermostat.applyThermostat(
        *_simulationBox,
        *_data,
        toTemperature(1e-6),
        0.1 * fs
    );

    EXPECT_TRUE(isfinite(_data->getNoseHooverMomentumEnergy()));
    EXPECT_TRUE(isfinite(_data->getNoseHooverFrictionEnergy()));
}

TEST_F(TestThermostat, langevinTinyTimestepSigmaStaysFinite)
{
    const auto sigma = thermostat::LangevinThermostat::calculateSigma(
        0.01 / s,
        toTemperature(300.0),
        0.1 * fs
    );

    EXPECT_TRUE(isfinite(sigma));
    EXPECT_GT(sigma.numerical_value_in(decltype(sigma)::unit), 0.0);
}

/* ---------- LangevinThermostat ---------- */

TEST_F(TestThermostat, langevinConstructorComputesSigma)
{
    // sigma > 0 once friction and targetTemp are non-zero.
    const auto sigma = thermostat::LangevinThermostat::calculateSigma(
        0.1 / s,
        toTemperature(300.0),
        0.1 * fs
    );
    EXPECT_GT(sigma.numerical_value_in(decltype(sigma)::unit), 0.0);
}

TEST_F(TestThermostat, langevinZeroFrictionHasZeroSigma)
{
    const auto sigma = thermostat::LangevinThermostat::calculateSigma(
        0.0 / s,
        toTemperature(300.0),
        0.1 * fs
    );

    EXPECT_DOUBLE_EQ(sigma.numerical_value_in(decltype(sigma)::unit), 0.0);
}

TEST_F(TestThermostat, langevinSigmaScalesWithTargetTemperature)
{
    const auto sigmaAt300 = thermostat::LangevinThermostat::calculateSigma(
        0.1 / s,
        toTemperature(300.0),
        0.1 * fs
    );
    const auto sigmaAt600 = thermostat::LangevinThermostat::calculateSigma(
        0.5 / s,
        toTemperature(600.0),
        0.1 * fs
    );

    // a higher target temperature must give a larger sigma (the
    // Langevin Gaussian-noise amplitude scales with sqrt(kBT)).
    EXPECT_NE(
        sigmaAt300.numerical_value_in(decltype(sigmaAt300)::unit),
        sigmaAt600.numerical_value_in(decltype(sigmaAt600)::unit)
    );
    EXPECT_GT(
        sigmaAt600.numerical_value_in(decltype(sigmaAt600)::unit),
        sigmaAt300.numerical_value_in(decltype(sigmaAt300)::unit)
    );
}

TEST_F(TestThermostat, langevinSigmaScalesWithFriction)
{
    const auto sigmaLow = thermostat::LangevinThermostat::calculateSigma(
        0.1 / s,
        toTemperature(300.0),
        0.1 * fs
    );
    const auto sigmaHigh = thermostat::LangevinThermostat::calculateSigma(
        0.5 / s,
        toTemperature(300.0),
        0.1 * fs
    );

    EXPECT_NE(
        sigmaLow.numerical_value_in(decltype(sigmaLow)::unit),
        sigmaHigh.numerical_value_in(decltype(sigmaHigh)::unit)
    );
    EXPECT_GT(
        sigmaHigh.numerical_value_in(decltype(sigmaHigh)::unit),
        sigmaLow.numerical_value_in(decltype(sigmaLow)::unit)
    );
}

TEST_F(TestThermostat, langevinThermostatType)
{
    auto langevin = thermostat::LangevinThermostat(0.1 / s);
    EXPECT_EQ(langevin.getThermostatType(), ThermostatType::LANGEVIN);
}

namespace
{
    constexpr size_t nBathAtoms = 2000;

    /**
     * @brief a box of non-interacting atoms of mass 12 with zero velocity
     */
    std::unique_ptr<molsys::SimulationBox> makeBath()
    {
        auto box = std::make_unique<molsys::SimulationBox>();

        for (size_t i = 0; i < nBathAtoms; ++i)
        {
            auto molecule = molsys::Molecule();
            auto atom     = std::make_shared<molsys::Atom>();

            atom->setMass(12.0 * g_per_mol);
            atom->setVelocity(linalg::Vec3D(0.0, 0.0, 0.0) * angstrom / s);
            molecule.addAtom(atom);

            box->addMolecule(molecule);
            box->addAtom(atom);
        }

        box->calculateDegreesOfFreedom();

        return box;
    }

    double temperatureOf(
        molsys::SimulationBox      &simulationBox,
        physicalData::PhysicalData &physicalData
    )
    {
        physicalData.calculateTemperature(simulationBox);
        return physicalData.getTemperature()
            .quantity_from_zero()
            .numerical_value_in(K);
    }
}   // namespace

TEST_F(TestThermostat, langevinWithoutFrictionLeavesVelocitiesUnchanged)
{
    auto langevin = thermostat::LangevinThermostat(0.0 / s);

    langevin.applyThermostatHalfStep(
        *_simulationBox,
        *_data,
        toTemperature(300.0),
        0.5 * fs
    );

    const auto &atoms = _simulationBox->getAtoms();
    EXPECT_EQ(
        atoms[0]->getVelocity(),
        linalg::Vec3D(1.0, 1.0, 1.0) * angstrom / s
    );
    EXPECT_EQ(
        atoms[1]->getVelocity(),
        linalg::Vec3D(1.0, 2.0, 3.0) * angstrom / s
    );
    EXPECT_EQ(
        atoms[2]->getVelocity(),
        linalg::Vec3D(1.0, 1.0, 1.0) * angstrom / s
    );
}

TEST_F(TestThermostat, langevinAtZeroTemperatureOnlyDampsVelocities)
{
    auto langevin = thermostat::LangevinThermostat(1.0e13 / s);

    langevin.applyThermostatHalfStep(
        *_simulationBox,
        *_data,
        toTemperature(0.0),
        0.5 * fs
    );

    for (const auto &atom : _simulationBox->getAtoms())
    {
        const Vel3D before = (atom == _simulationBox->getAtoms()[1])
                                 ? linalg::Vec3D(1.0, 2.0, 3.0) * angstrom / s
                                 : linalg::Vec3D(1.0, 1.0, 1.0) * angstrom / s;

        // the damping shrinks every component towards zero, never flips it
        for (size_t axis = 0; axis < 3; ++axis)
        {
            EXPECT_GT(atom->getVelocity()[axis], Vel::zero());
            EXPECT_LT(atom->getVelocity()[axis], before[axis]);
        }
    }
}

TEST_F(TestThermostat, langevinSameSeedGivesSameVelocities)
{
    auto run = []()
    {
        settings::GeneralSettings::setRandomSeed(42);
        settings::GeneralSettings::setIsRandomSeedSet(true);

        auto       bath     = makeBath();
        auto       langevin = thermostat::LangevinThermostat(1.0e12 / s);
        const auto target   = toTemperature(300.0);

        for (size_t step = 0; step < 5; ++step)
        {
            langevin.applyLangevin(*bath, target, 0.5 * fs);
        }

        settings::GeneralSettings::setIsRandomSeedSet(false);

        return bath->getAtoms()[0]->getVelocity();
    };

    const auto first  = run();
    const auto second = run();

    EXPECT_EQ(first, second);
    EXPECT_NE(first, linalg::Vec3D(0.0, 0.0, 0.0) * angstrom / s);
}

TEST_F(TestThermostat, langevinBathEquilibratesToTargetTemperature)
{
    settings::GeneralSettings::setRandomSeed(7);
    settings::GeneralSettings::setIsRandomSeedSet(true);

    auto bath     = makeBath();
    auto data     = physicalData::PhysicalData();
    auto langevin = thermostat::LangevinThermostat(1.0e13 / s);
    auto target   = toTemperature(300.0);

    settings::GeneralSettings::setIsRandomSeedSet(false);

    for (size_t step = 0; step < 4000; ++step)
    {
        langevin.applyLangevin(*bath, target, 0.5 * fs);
    }

    double sum = 0.0;
    for (size_t step = 0; step < 200; ++step)
    {
        langevin.applyLangevin(*bath, target, 0.5 * fs);
        sum += temperatureOf(*bath, data);
    }

    // 2000 atoms: the temperature of one frame fluctuates by about 1.3 %
    // (sqrt(2 / (3 N))), the mean over many frames is much tighter
    EXPECT_NEAR(sum / 200.0, 300.0, 6.0);
}

TEST_F(TestThermostat, langevinFullStepUpdatesTemperatureOfPhysicalData)
{
    auto langevin = thermostat::LangevinThermostat(1.0e13 / s);

    langevin.applyThermostat(
        *_simulationBox,
        *_data,
        toTemperature(300.0),
        0.5 * fs
    );

    const auto temperature =
        _data->getTemperature().quantity_from_zero().numerical_value_in(K);

    EXPECT_TRUE(std::isfinite(temperature));
    EXPECT_GT(temperature, 0.0);
}

TEST_F(TestThermostat, langevinCopyKeepsFriction)
{
    settings::GeneralSettings::setRandomSeed(3);
    settings::GeneralSettings::setIsRandomSeedSet(true);

    auto original = thermostat::LangevinThermostat(1.0e13 / s);
    auto copied   = thermostat::LangevinThermostat(original);
    auto assigned = thermostat::LangevinThermostat(0.0 / s);
    assigned      = original;

    settings::GeneralSettings::setIsRandomSeedSet(false);

    const auto target = toTemperature(300.0);
    auto       boxA   = makeBath();
    auto       boxB   = makeBath();
    auto       data   = physicalData::PhysicalData();

    for (size_t step = 0; step < 2000; ++step)
    {
        copied.applyLangevin(*boxA, target, 0.5 * fs);
        assigned.applyLangevin(*boxB, target, 0.5 * fs);
    }

    // a copy with the friction of zero would stay at zero temperature
    EXPECT_GT(temperatureOf(*boxA, data), 100.0);
    EXPECT_GT(temperatureOf(*boxB, data), 100.0);
}

/* ---------- NoseHooverThermostat ---------- */

TEST_F(TestThermostat, noseHooverThermostatType)
{
    NoseHooverSettings noseHooverSettings;
    noseHooverSettings.setNoseHooverCouplingFrequencyByUser(1.0e13 * Hz);
    auto thermostat = thermostat::NoseHooverThermostat(noseHooverSettings);
    EXPECT_EQ(thermostat.getThermostatType(), ThermostatType::NOSE_HOOVER);
}

TEST_F(TestThermostat, noseHooverAppliesFiniteForceAndStateUpdates)
{
    NoseHooverSettings noseHooverSettings;
    noseHooverSettings.setNoseHooverCouplingFrequencyByUser(1.0e13 * Hz);
    const auto chis = std::map<size_t, Action>{
        {0, 0.1 * kcal_per_mol * s},
        {1, 0.2 * kcal_per_mol * s},
        {2, 0.3 * kcal_per_mol * s}
    };
    const auto zetas = std::map<size_t, Quantity>{{0, 0.0}, {1, 0.0}, {2, 0.0}};
    noseHooverSettings.setChiByUser(chis);
    noseHooverSettings.setZetaByUser(zetas);
    auto thermostat = thermostat::NoseHooverThermostat(noseHooverSettings);

    thermostat.applyThermostatOnForces(*_simulationBox, toTemperature(300.0));
    for (const auto &atom : _simulationBox->getAtoms())
        for (size_t axis = 0; axis < 3; ++axis)
            EXPECT_TRUE(isfinite(atom->getForce()[axis]));

    const auto chiBefore  = thermostat.getChi();
    const auto zetaBefore = thermostat.getZeta();
    thermostat.applyThermostat(
        *_simulationBox,
        *_data,
        toTemperature(300.0),
        0.1 * fs
    );

    EXPECT_TRUE(
        std::isfinite(
            _data->getTemperature().quantity_from_zero().numerical_value_in(K)
        )
    );
    EXPECT_TRUE(isfinite(_data->getNoseHooverMomentumEnergy()));
    EXPECT_TRUE(isfinite(_data->getNoseHooverFrictionEnergy()));
    EXPECT_NE(thermostat.getChi(), chiBefore);
    EXPECT_NE(thermostat.getZeta(), zetaBefore);
}

/* ---------- additional value-level coverage ---------- */

TEST_F(TestThermostat, temperatureSetterGetter)
{
    _thermostat->setTemperature(mstd::units::point<K>(123.0));
    EXPECT_EQ(_thermostat->getTemperature(), mstd::units::point<K>(123.0));
}

TEST_F(TestThermostat, berendsenStoresMeasuredTemperature)
{
    _thermostat = std::make_unique<thermostat::BerendsenThermostat>(100.0 * fs);

    _data->calculateTemperature(*_simulationBox);
    const auto oldTemperature = _data->getTemperature();

    _thermostat->applyThermostat(
        *_simulationBox,
        *_data,
        toTemperature(300.0),
        0.1 * fs
    );

    EXPECT_EQ(_thermostat->getTemperature(), oldTemperature);
}

TEST_F(TestThermostat, berendsenScalesVelocitiesBySqrtOfTemperatureRatio)
{
    _thermostat = std::make_unique<thermostat::BerendsenThermostat>(100.0 * fs);

    _data->calculateTemperature(*_simulationBox);
    const auto oldTemperature = _data->getTemperature();
    const auto oldVelocity =
        _simulationBox->getAtoms()[1]->getVelocity().numerical_value_in(
            angstrom / s
        );

    _thermostat->applyThermostat(
        *_simulationBox,
        *_data,
        toTemperature(300.0),
        0.1 * fs
    );

    const auto factor =
        std::sqrt((_data->getTemperature().quantity_from_zero() /
                   oldTemperature.quantity_from_zero())
                      .numerical_value_in(one));
    const auto newVelocity =
        _simulationBox->getAtoms()[1]->getVelocity().numerical_value_in(
            angstrom / s
        );
    for (size_t axis = 0; axis < 3; ++axis)
        EXPECT_NEAR(newVelocity[axis], factor * oldVelocity[axis], 1e-12);
}

namespace
{
    std::vector<Vel3D> saveVelocities(
        const molsys::SimulationBox &simulationBox
    )
    {
        std::vector<Vel3D> velocities;
        for (const auto &atom : simulationBox.getAtoms())
            velocities.push_back(atom->getVelocity());
        return velocities;
    }

    void restoreVelocities(
        molsys::SimulationBox    &simulationBox,
        const std::vector<Vel3D> &velocities
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
    _data->calculateTemperature(*_simulationBox);
    const auto temperature = _data->getTemperature();
    const auto initial     = saveVelocities(*_simulationBox);
    const auto dof = static_cast<double>(_simulationBox->getDegreesOfFreedom());

    const auto tempRatio = 10.0;
    const auto timeRatio = 0.1 / 10.0;

    auto       thermostat = thermostat::VelocityRescalingThermostat(10.0 * fs);
    const auto target =
        Temperature{tempRatio * temperature.quantity_from_zero()};

    const auto expectedMean = 1.0 + (timeRatio * (tempRatio - 1.0));
    const auto expectedStd  = 2.0 * std::sqrt(timeRatio * tempRatio / dof);

    constexpr size_t nSamples = 4000;
    double           sum      = 0.0;
    double           sumSq    = 0.0;

    for (size_t i = 0; i < nSamples; ++i)
    {
        restoreVelocities(*_simulationBox, initial);
        thermostat.applyThermostat(*_simulationBox, *_data, target, 0.1 * fs);

        const auto lambda = (_data->getTemperature().quantity_from_zero() /
                             temperature.quantity_from_zero())
                                .numerical_value_in(one);
        sum   += lambda;
        sumSq += lambda * lambda;

        const auto scaled = _simulationBox->getAtoms()[1]->getVelocity();
        for (size_t axis = 0; axis < 3; ++axis)
        {
            ASSERT_NEAR(
                scaled[axis].numerical_value_in(angstrom / s),
                std::sqrt(lambda) *
                    initial[1][axis].numerical_value_in(angstrom / s),
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
    _data->calculateTemperature(*_simulationBox);
    const auto temperature = _data->getTemperature();

    auto thermostat = thermostat::VelocityRescalingThermostat(100.0 * fs);
    thermostat.applyThermostat(
        *_simulationBox,
        *_data,
        Temperature{3.0 * temperature.quantity_from_zero()},
        1e-9 * fs
    );

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
    const auto targetTemp = 300.0;
    const auto freq       = 1.0;
    auto       chi        = std::vector<double>{0.1, 0.2, 0.3};
    auto       zeta       = std::vector<double>{0.0, 0.0, 0.0};

    NoseHooverSettings noseHooverSettings;
    noseHooverSettings.setNoseHooverChainLengthByUser(chi.size() - 1);
    noseHooverSettings.setNoseHooverCouplingFrequencyByUser(freq * Hz);
    noseHooverSettings.setChiByUser(
        {{0, 0.1 * kcal_per_mol * s},
         {1, 0.2 * kcal_per_mol * s},
         {2, 0.3 * kcal_per_mol * s}}
    );
    noseHooverSettings.setZetaByUser({{0, 0.0}, {1, 0.0}, {2, 0.0}});
    auto thermostat = thermostat::NoseHooverThermostat(noseHooverSettings);

    thermostat.applyThermostat(
        *_simulationBox,
        *_data,
        toTemperature(targetTemp),
        0.1 * fs
    );

    const auto temperature = _data->getTemperature();
    const auto dof = static_cast<double>(_simulationBox->getDegreesOfFreedom());
    const auto timeStep = 0.1e-15;
    const auto freq2    = freq * freq;
    const auto kineticEnergy =
        (1.0 * gas_constant).numerical_value_in(kcal_per_mol / K) *
        temperature.quantity_from_zero().numerical_value_in(K);
    const auto eTar =
        (1.0 * gas_constant).numerical_value_in(kcal_per_mol / K) * targetTemp;

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

    // the last reference entry is the hidden boundary term
    ASSERT_EQ(thermostat.getChi().size(), chi.size() - 1);
    ASSERT_EQ(thermostat.getZeta().size(), zeta.size() - 1);
    for (size_t i = 0; i < chi.size() - 1; ++i)
    {
        EXPECT_DOUBLE_EQ(
            thermostat.getChi()[i].numerical_value_in(kcal_per_mol * s),
            chi[i]
        );
        EXPECT_DOUBLE_EQ(
            thermostat.getZeta()[i].numerical_value_in(one),
            zeta[i]
        );
    }
    EXPECT_DOUBLE_EQ(
        _data->getNoseHooverMomentumEnergy().numerical_value_in(kcal_per_mol),
        eMom
    );
    EXPECT_DOUBLE_EQ(
        _data->getNoseHooverFrictionEnergy().numerical_value_in(one),
        eFri
    );
    EXPECT_EQ(thermostat.getTemperature(), temperature);
}

TEST_F(TestThermostat, noseHooverForceMatchesReference)
{
    const auto targetTemp = 250.0;
    const auto chi0       = 0.4;
    const auto freq       = 2.0;

    NoseHooverSettings noseHooverSettings;
    noseHooverSettings.setNoseHooverCouplingFrequencyByUser(freq * Hz);
    const auto chis = std::map<size_t, Action>{
        {0, chi0 * kcal_per_mol * s},
        {1, 0.2 * kcal_per_mol * s},
        {2, 0.3 * kcal_per_mol * s}
    };
    const auto zetas = std::map<size_t, Quantity>{
        {0, 0.0 * one},
        {1, 0.0 * one},
        {2, 0.0 * one}
    };
    noseHooverSettings.setChiByUser(chis);
    noseHooverSettings.setZetaByUser(zetas);
    auto thermostat = thermostat::NoseHooverThermostat(noseHooverSettings);

    for (auto &atom : _simulationBox->getAtoms())
        atom->setForce(Force3D::zero());

    thermostat.applyThermostatOnForces(
        *_simulationBox,
        toTemperature(targetTemp)
    );

    const auto dof = static_cast<double>(_simulationBox->getDegreesOfFreedom());
    const auto thermalEnergy =
        (1.0 * gas_constant).numerical_value_in(kcal_per_mol / K) * targetTemp *
        kcal_per_mol;
    const auto factor =
        (chis.at(0) * (freq * Hz) * (freq * Hz)) / (thermalEnergy * dof);

    for (const auto &atom : _simulationBox->getAtoms())
    {
        for (size_t axis = 0; axis < 3; ++axis)
        {
            const auto momentum = atom->getMass() * atom->getVelocity()[axis];
            const auto expected = -factor * momentum;

            EXPECT_DOUBLE_EQ(
                atom->getForce()[axis].numerical_value_in(
                    kcal_per_mol / angstrom
                ),
                expected.numerical_value_in(kcal_per_mol / angstrom)
            );
        }
    }
}

TEST_F(TestThermostat, noseHooverRejectsNegativeTargetTemperature)
{
    const auto nhSettings = noseHooverSettings();
    auto       thermostat = noseHoover(nhSettings);

    EXPECT_THROW_MSG(
        thermostat.applyThermostat(
            *_simulationBox,
            *_data,
            toTemperature(-5.0),
            0.1 * fs
        ),
        exc::UserInputException,
        NOSE_HOOVER_ZERO_TARGET
    );
}

TEST_F(TestThermostat, langevinSigmaMatchesReference)
{
    // friction is given in 1/ps in the input file
    const auto friction = 0.1 / ps;
    const auto sigma    = thermostat::LangevinThermostat::calculateSigma(
        friction,
        toTemperature(300.0),
        0.1 * fs
    );

    // independent SI reference: sigma^2 = 4 * gamma * R * T / dt
    constexpr double gasConstantSI = 8.314462618;   // J / (mol K)
    const auto       gammaSI = (friction * s).numerical_value_in(mp_units::one);
    const auto       dtSI    = (0.1 * fs).numerical_value_in(s);
    const auto       expected =
        4.0 * gammaSI * gasConstantSI * 300.0 / dtSI / 4184.0;   // kcal/mol/s^2

    EXPECT_NEAR(
        (sigma * sigma).numerical_value_in(kcal_per_mol / s / s),
        expected,
        1e-9 * expected
    );

    // sigma scales with sqrt(T)
    const auto sigma4 = thermostat::LangevinThermostat::calculateSigma(
        friction,
        toTemperature(1200.0),
        0.1 * fs
    );
    const auto sigmaValue = sigma.numerical_value_in(decltype(sigma)::unit);
    EXPECT_NEAR(
        sigma4.numerical_value_in(decltype(sigma4)::unit),
        2.0 * sigmaValue,
        1e-12 * sigmaValue
    );
}
