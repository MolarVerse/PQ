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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "atom.hpp"
#include "exceptions.hpp"
#include "maxwellBoltzmann.hpp"
#include "settings.hpp"
#include "simulationBox.hpp"
#include "thermostatSettings.hpp"
#include "throwWithMessage.hpp"
#include "vector3d.hpp"   // IWYU pragma: keep

namespace
{
    using linearAlgebra::Vec3D;

    // Build a 3x3x3 block of 27 atoms with different masses on a slightly
    // distorted lattice (deterministic - no random numbers needed).
    molsys::SimulationBox *makeBox()
    {
        auto *box = new molsys::SimulationBox();

        const std::vector<double> masses = {1.0, 12.0, 14.0, 16.0};

        auto totalMass = 0.0;
        for (size_t i = 0; i < 27; ++i)
        {
            const auto x = static_cast<double>(i % 3);
            const auto y = static_cast<double>((i / 3) % 3);
            const auto z = static_cast<double>(i / 9);
            const auto d = 0.1 * std::sin(static_cast<double>(i));

            auto atom = std::make_shared<molsys::Atom>();
            atom->setMass(masses[i % masses.size()]);
            atom->setPosition(Vec3D(1.5 * x + d, 1.5 * y - d, 1.5 * z + 0.5 * d)
            );
            atom->setVelocity(Vec3D(0.0, 0.0, 0.0));

            totalMass += atom->getMass();
            box->addAtom(atom);
        }

        box->setTotalMass(totalMass);
        box->calculateDegreesOfFreedom();
        box->calculateCenterOfMass();

        return box;
    }

    // fixes the random seed for the lifetime of the guard
    struct SeedGuard
    {
        explicit SeedGuard(uint_fast32_t seed)
        {
            settings::Settings::setRandomSeed(seed);
            settings::Settings::setIsRandomSeedSet(true);
        }

        ~SeedGuard() { settings::Settings::setIsRandomSeedSet(false); }

        SeedGuard(const SeedGuard &)            = delete;
        SeedGuard &operator=(const SeedGuard &) = delete;
    };

    std::vector<Vec3D> velocitiesOf(molsys::SimulationBox &box)
    {
        std::vector<Vec3D> velocities;
        for (const auto &atom : box.getAtoms())
            velocities.push_back(atom->getVelocity());
        return velocities;
    }

    // sum of |m v| - natural scale of the total linear momentum
    double momentumScale(molsys::SimulationBox &box)
    {
        auto scale = 0.0;
        for (const auto &atom : box.getAtoms())
            scale += atom->getMass() * norm(atom->getVelocity());
        return scale;
    }

    // sum of |m (r - R)| |v| - natural scale of the total angular momentum
    double angularMomentumScale(molsys::SimulationBox &box)
    {
        box.calculateCenterOfMass();
        const auto centerOfMass = box.getCenterOfMass();

        auto scale = 0.0;
        for (const auto &atom : box.getAtoms())
        {
            const auto relativePosition = atom->getPosition() - centerOfMass;
            scale += atom->getMass() * norm(relativePosition) *
                     norm(atom->getVelocity());
        }
        return scale;
    }

    Vec3D angularMomentumOf(molsys::SimulationBox &box)
    {
        box.calculateCenterOfMass();
        return box.calculateAngularMomentum(box.calculateMomentum());
    }
}   // namespace

TEST(TestMaxwellBoltzmann, initializeVelocitiesRemovesLinearMomentum)
{
    settings::ThermostatSettings::setTargetTemperature(300.0);

    auto *box = makeBox();
    maxwellBoltzmann::MaxwellBoltzmann().initializeVelocities(*box);

    const auto momentum = box->calculateMomentum();
    const auto scale    = momentumScale(*box);
    ASSERT_GT(scale, 0.0);

    for (size_t i = 0; i < 3; ++i) EXPECT_NEAR(momentum[i], 0.0, 1e-10 * scale);

    delete box;
}

TEST(TestMaxwellBoltzmann, initializeVelocitiesRemovesAngularMomentum)
{
    settings::ThermostatSettings::setTargetTemperature(300.0);

    auto *box = makeBox();
    maxwellBoltzmann::MaxwellBoltzmann().initializeVelocities(*box);

    const auto angularMomentum = angularMomentumOf(*box);
    const auto scale           = angularMomentumScale(*box);
    ASSERT_GT(scale, 0.0);

    for (size_t i = 0; i < 3; ++i)
        EXPECT_NEAR(angularMomentum[i], 0.0, 1e-10 * scale);

    delete box;
}

TEST(TestMaxwellBoltzmann, initializeVelocitiesReachesTargetTemperature)
{
    for (const auto target : {10.0, 300.0, 5000.0})
    {
        SCOPED_TRACE("target temperature " + std::to_string(target));

        settings::ThermostatSettings::setTargetTemperature(target);

        auto *box = makeBox();
        maxwellBoltzmann::MaxwellBoltzmann().initializeVelocities(*box);

        // temperature is rescaled last and rescaling does not reintroduce
        // any linear or angular momentum
        EXPECT_NEAR(box->calculateTemperature(), target, 1e-9 * target);

        delete box;
    }
}

TEST(TestMaxwellBoltzmann, initializeVelocitiesSatisfiesAllConstraintsTogether)
{
    settings::ThermostatSettings::setTargetTemperature(300.0);

    // different random numbers must never break the ordering
    // momentum -> angular momentum -> temperature
    for (uint_fast32_t seed = 1; seed <= 25; ++seed)
    {
        SCOPED_TRACE("seed " + std::to_string(seed));
        const SeedGuard guard(seed);

        auto *box = makeBox();
        maxwellBoltzmann::MaxwellBoltzmann().initializeVelocities(*box);

        const auto momentum        = box->calculateMomentum();
        const auto angularMomentum = angularMomentumOf(*box);
        const auto pScale          = momentumScale(*box);
        const auto lScale          = angularMomentumScale(*box);

        for (size_t i = 0; i < 3; ++i)
        {
            EXPECT_NEAR(momentum[i], 0.0, 1e-10 * pScale);
            EXPECT_NEAR(angularMomentum[i], 0.0, 1e-10 * lScale);
        }
        EXPECT_NEAR(box->calculateTemperature(), 300.0, 1e-9 * 300.0);

        delete box;
    }
}

TEST(TestMaxwellBoltzmann, initializeVelocitiesGivesFiniteNonZeroVelocities)
{
    settings::ThermostatSettings::setTargetTemperature(300.0);

    auto *box = makeBox();
    maxwellBoltzmann::MaxwellBoltzmann().initializeVelocities(*box);

    for (const auto &atom : box->getAtoms())
    {
        const auto velocity = atom->getVelocity();
        EXPECT_GT(norm(velocity), 0.0);
        for (size_t i = 0; i < 3; ++i) EXPECT_TRUE(std::isfinite(velocity[i]));
    }

    delete box;
}

TEST(
    TestMaxwellBoltzmann,
    initializeVelocitiesLeavesPositionsAndMassesUntouched
)
{
    settings::ThermostatSettings::setTargetTemperature(300.0);

    auto *reference = makeBox();
    auto *box       = makeBox();

    maxwellBoltzmann::MaxwellBoltzmann().initializeVelocities(*box);

    ASSERT_EQ(box->getNumberOfAtoms(), reference->getNumberOfAtoms());
    for (size_t i = 0; i < box->getNumberOfAtoms(); ++i)
    {
        EXPECT_EQ(
            box->getAtoms()[i]->getPosition(),
            reference->getAtoms()[i]->getPosition()
        );
        EXPECT_DOUBLE_EQ(
            box->getAtoms()[i]->getMass(),
            reference->getAtoms()[i]->getMass()
        );
    }
    EXPECT_DOUBLE_EQ(box->getTotalMass(), reference->getTotalMass());

    delete reference;
    delete box;
}

TEST(TestMaxwellBoltzmann, initializeVelocitiesIsReproducibleForFixedSeed)
{
    settings::ThermostatSettings::setTargetTemperature(300.0);

    std::vector<Vec3D> first;
    std::vector<Vec3D> second;
    std::vector<Vec3D> other;

    {
        const SeedGuard guard(4711);
        auto           *box = makeBox();
        maxwellBoltzmann::MaxwellBoltzmann().initializeVelocities(*box);
        first = velocitiesOf(*box);
        delete box;
    }
    {
        const SeedGuard guard(4711);
        auto           *box = makeBox();
        maxwellBoltzmann::MaxwellBoltzmann().initializeVelocities(*box);
        second = velocitiesOf(*box);
        delete box;
    }
    {
        const SeedGuard guard(4712);
        auto           *box = makeBox();
        maxwellBoltzmann::MaxwellBoltzmann().initializeVelocities(*box);
        other = velocitiesOf(*box);
        delete box;
    }

    EXPECT_EQ(first, second);
    EXPECT_NE(first, other);
}

TEST(TestMaxwellBoltzmann, initializeVelocitiesForZeroTargetTemperatureThrows)
{
    // a zero target temperature generates only zero velocities, which cannot
    // be rescaled to a (zero) target - the final temperature reset rejects it
    settings::ThermostatSettings::setTargetTemperature(0.0);

    auto *box = makeBox();

    EXPECT_THROW_MSG(
        maxwellBoltzmann::MaxwellBoltzmann().initializeVelocities(*box),
        exc::UserInputException,
        "Cannot rescale a zero-temperature system. Initialize velocities first."
    );

    delete box;
}
