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

#include <gtest/gtest.h>

#include "exceptions.hpp"
#include "kinetics.hpp"
#include "thermostatSettings.hpp"
#include "throwWithMessage.hpp"
#include "timingsSettings.hpp"
#include "unit_types.hpp"
#include "units.hpp"

namespace
{
    KineticsSettings rampSettings(
        const double start,
        const double end,
        const size_t steps,
        const size_t frequency
    )
    {
        KineticsSettings settings;
        settings.setStartTemperatureByUser(toTemperature(start));
        settings.setEndTemperatureByUser(toTemperature(end));
        settings.setRampingStepsByUser(steps);
        settings.setRampingFrequencyByUser(frequency);
        settings.finalize();
        return settings;
    }

    void expectTemperature(const kinetics::Kinetics &kin, const double kelvin)
    {
        EXPECT_DOUBLE_EQ(
            kin.getCurrentTargetTemperature()
                .quantity_from_zero()
                .numerical_value_in(K),
            kelvin
        );
    }
}   // namespace

TEST(TestKinetics, noRampKeepsTargetTemperature)
{
    KineticsSettings settings;
    settings.setTargetTemperatureByUser(toTemperature(300.0));
    settings.finalize();

    kinetics::Kinetics kin{settings};
    expectTemperature(kin, 300.0);

    kin.applyTemperatureRamping();
    expectTemperature(kin, 300.0);
}

TEST(TestKinetics, startsAtStartTemperature)
{
    const auto settings = rampSettings(300.0, 302.0, 2, 1);

    kinetics::Kinetics kin{settings};
    expectTemperature(kin, 300.0);
}

TEST(TestKinetics, rampEveryStepReachesEndTemperatureAndStops)
{
    const auto         settings = rampSettings(300.0, 302.0, 2, 1);
    kinetics::Kinetics kin{settings};

    kin.applyTemperatureRamping();
    expectTemperature(kin, 301.0);

    kin.applyTemperatureRamping();
    expectTemperature(kin, 302.0);

    // ramp is finished, the temperature must stay constant
    kin.applyTemperatureRamping();
    expectTemperature(kin, 302.0);
}

TEST(TestKinetics, rampWithFrequencyUpdatesEveryNthStep)
{
    const auto         settings = rampSettings(300.0, 302.0, 4, 2);
    kinetics::Kinetics kin{settings};

    kin.applyTemperatureRamping();
    expectTemperature(kin, 300.0);
    kin.applyTemperatureRamping();
    expectTemperature(kin, 301.0);
    kin.applyTemperatureRamping();
    expectTemperature(kin, 301.0);
    kin.applyTemperatureRamping();
    expectTemperature(kin, 302.0);
    kin.applyTemperatureRamping();
    expectTemperature(kin, 302.0);
}

TEST(TestKinetics, onlyEndTemperatureDefinedIsUsedFromStart)
{
    KineticsSettings kineticsSettings;
    kineticsSettings.setEndTemperatureByUser(toTemperature(300.0));
    kineticsSettings.finalize();

    kinetics::Kinetics kin{kineticsSettings};
    expectTemperature(kin, 300.0);
}

TEST(TestKinetics, finalizeComputesIncreaseAndStepsForFullSimulation)
{
    settings::TimingsSettings::setNumberOfSteps(100);

    KineticsSettings kineticsSettings;
    kineticsSettings.setTargetTemperatureByUser(toTemperature(300.0));
    kineticsSettings.setStartTemperatureByUser(toTemperature(200.0));
    kineticsSettings.finalize();

    EXPECT_EQ(
        kineticsSettings.getTemperatureIncrease(),
        mstd::units::delta<K>(1.0)
    );
    EXPECT_EQ(kineticsSettings.getRampingSteps(), 100);
}

TEST(TestKinetics, finalizeComputesIncreaseForExplicitSteps)
{
    settings::TimingsSettings::setNumberOfSteps(100);

    KineticsSettings kineticsSettings;
    kineticsSettings.setTargetTemperatureByUser(toTemperature(300.0));
    kineticsSettings.setStartTemperatureByUser(toTemperature(200.0));
    kineticsSettings.setRampingStepsByUser(50);
    kineticsSettings.finalize();

    EXPECT_EQ(
        kineticsSettings.getTemperatureIncrease(),
        mstd::units::delta<K>(2.0)
    );
    EXPECT_EQ(kineticsSettings.getRampingSteps(), 50);
}

TEST(TestKinetics, finalizeComputesIncreaseWithRampingFrequency)
{
    settings::TimingsSettings::setNumberOfSteps(100);

    KineticsSettings kineticsSettings;
    kineticsSettings.setTargetTemperatureByUser(toTemperature(300.0));
    kineticsSettings.setStartTemperatureByUser(toTemperature(200.0));
    kineticsSettings.setRampingFrequencyByUser(2);
    kineticsSettings.finalize();

    EXPECT_EQ(
        kineticsSettings.getTemperatureIncrease(),
        mstd::units::delta<K>(2.0)
    );
    EXPECT_EQ(kineticsSettings.getRampingSteps(), 100);
    EXPECT_EQ(kineticsSettings.getRampingFrequency(), 2);
}

TEST(TestKinetics, rampReachesEndWithPartialFinalInterval)
{
    settings::TimingsSettings::setNumberOfSteps(10);
    const auto kineticsSettings = rampSettings(200.0, 300.0, 10, 3);

    // ceil(10 / 3) = 4 updates
    EXPECT_EQ(
        kineticsSettings.getTemperatureIncrease(),
        mstd::units::delta<K>(25.0)
    );

    kinetics::Kinetics kin{kineticsSettings};
    for (size_t step = 0; step < 10; ++step) kin.applyTemperatureRamping();

    expectTemperature(kin, 300.0);
}

TEST(TestKinetics, rejectsEmptyTemperatureRamp)
{
    settings::TimingsSettings::setNumberOfSteps(0);

    KineticsSettings kineticsSettings;
    kineticsSettings.setTargetTemperatureByUser(toTemperature(300.0));
    kineticsSettings.setStartTemperatureByUser(toTemperature(200.0));

    EXPECT_THROW_MSG(
        kineticsSettings.finalize(),
        exc::InputFileException,
        "Temperature ramp requires at least one simulation step"
    );
}

TEST(TestKinetics, rejectsZeroTemperatureRampFrequency)
{
    settings::TimingsSettings::setNumberOfSteps(10);

    KineticsSettings kineticsSettings;
    kineticsSettings.setTargetTemperatureByUser(toTemperature(300.0));
    kineticsSettings.setStartTemperatureByUser(toTemperature(200.0));
    kineticsSettings.setRampingStepsByUser(10);
    kineticsSettings.setRampingFrequencyByUser(0);

    EXPECT_THROW_MSG(
        kineticsSettings.finalize(),
        exc::InputFileException,
        "Temperature ramp frequency must be greater than zero"
    );
}

TEST(TestKinetics, rejectsStartTemperatureWithoutEndTemperature)
{
    KineticsSettings kineticsSettings;
    kineticsSettings.setStartTemperatureByUser(toTemperature(200.0));

    EXPECT_THROW_MSG(
        kineticsSettings.finalize(),
        exc::InputFileException,
        "A start temperature requires an end temperature or a target "
        "temperature to be defined"
    );
}
