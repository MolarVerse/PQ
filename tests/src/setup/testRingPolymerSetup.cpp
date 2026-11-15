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

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <memory>
#include <typeinfo>
#include <vector>

#include "atom.hpp"
#include "fileSettings.hpp"
#include "generalSettings.hpp"
#include "molecule.hpp"
#include "ringPolymerSettings.hpp"
#include "ringPolymerSetup.hpp"
#include "ringPolymerqmmdEngine.hpp"
#include "simulationBox.hpp"
#include "testSetup.hpp"
#include "unit_types.hpp"
#include "units.hpp"

TEST_F(TestSetup, setupRingPolymerIsNoOpWhenNotActivated)
{
    settings::GeneralSettings::setIsRingPolymerMDActivated(false);
    EXPECT_NO_THROW(setup::setupRingPolymer(*_engine));
}

TEST_F(TestSetup, ringPolymerSetupPhysicalDataResizesBeads)
{
    Settings settings;
    settings.ringPolymerSettings.setNumberOfBeadsByUser(4);
    engine::RingPolymerQMMDEngine rpEngine{settings};

    setup::RingPolymerSetup setup(rpEngine);
    EXPECT_NO_THROW(setup.setupPhysicalData());
}

TEST_F(TestSetup, ringPolymerSetupSimulationBoxAddsBeadsToEngine)
{
    Settings settings;
    settings.ringPolymerSettings.setNumberOfBeadsByUser(3);
    engine::RingPolymerQMMDEngine rpEngine{settings};

    setup::RingPolymerSetup setup(rpEngine);
    EXPECT_NO_THROW(setup.setupSimulationBox());
}

namespace
{
    constexpr size_t N_TEST_ATOMS = 600;

    /**
     * @brief fills the simulation box of the engine with non-bonded atoms of
     * different masses at distinct positions, one atom per molecule
     */
    void fillEngineBox(engine::RingPolymerQMMDEngine &ringPolymerEngine)
    {
        auto &simulationBox = ringPolymerEngine.getSimulationBox();

        for (size_t index = 0; index < N_TEST_ATOMS; ++index)
        {
            auto molecule = molsys::Molecule();
            auto atom     = std::make_shared<molsys::Atom>();

            atom->setMass(1.0 + (static_cast<double>(index % 4) * 5.0));
            const auto value = static_cast<double>(index);
            atom->setPosition(
                {value, 2.0 * std::sin(value), 3.0 * std::cos(0.7 * value)}
            );
            atom->setVelocity({0.0, 0.0, 0.0});
            molecule.addAtom(atom);

            simulationBox.addMolecule(molecule);
            simulationBox.addAtom(atom);
        }

        simulationBox.calculateDegreesOfFreedom();
        simulationBox.calculateTotalMass();
        simulationBox.calculateCenterOfMass();
    }

    double temperatureOf(molsys::SimulationBox &simulationBox)
    {
        return simulationBox.calculateTemperature()
            .quantity_from_zero()
            .numerical_value_in(K);
    }
}   // namespace

/**
 * @brief the physical data of every bead is created
 */
TEST_F(TestSetup, ringPolymerSetupPhysicalDataHasOneEntryPerBead)
{
    Settings settings;
    settings.ringPolymerSettings.setNumberOfBeadsByUser(4);
    engine::RingPolymerQMMDEngine rpEngine{settings};

    setup::RingPolymerSetup setup(rpEngine);
    setup.setupPhysicalData();

    EXPECT_EQ(rpEngine.getRingPolymerBeadsPhysicalData().size(), 4);
}

/**
 * @brief every bead is an independent copy of the simulation box
 */
TEST_F(TestSetup, ringPolymerSetupBeadsAreIndependentCopiesOfTheBox)
{
    Settings settings;
    settings.ringPolymerSettings.setNumberOfBeadsByUser(3);
    engine::RingPolymerQMMDEngine rpEngine{settings};
    fillEngineBox(rpEngine);

    setup::RingPolymerSetup setup(rpEngine);
    setup.setupSimulationBox();

    auto &beads = rpEngine.getRingPolymerBeads();
    ASSERT_EQ(beads.size(), 3);

    for (auto &bead : beads)
    {
        ASSERT_EQ(bead.getNumberOfAtoms(), N_TEST_ATOMS);
        EXPECT_EQ(
            bead.getPositions(),
            rpEngine.getSimulationBox().getPositions()
        );
    }

    beads[0].getAtoms()[5]->setPosition({-1.0, -2.0, -3.0});

    EXPECT_EQ(
        beads[1].getAtoms()[5]->getPosition(),
        linalg::Vec3D(5, 2.0 * std::sin(5.0), 3.0 * std::cos(3.5))
    );
    EXPECT_EQ(
        rpEngine.getSimulationBox().getAtoms()[5]->getPosition(),
        linalg::Vec3D(5, 2.0 * std::sin(5.0), 3.0 * std::cos(3.5))
    );
    EXPECT_NE(beads[0].getAtoms()[5], beads[1].getAtoms()[5]);
}

/**
 * @brief without a restart file every bead gets its own Maxwell-Boltzmann
 * velocities at exactly the start temperature, and no net momentum; with a
 * fixed random seed the beads differ from each other, and a second setup with
 * the same seed gives the same velocities
 */
TEST_F(TestSetup, ringPolymerSetupInitializesVelocitiesAtStartTemperature)
{
    settings::GeneralSettings::setRandomSeed(11);
    settings::GeneralSettings::setIsRandomSeedSet(true);

    Settings settings;
    settings.ringPolymerSettings.setNumberOfBeadsByUser(3);
    settings.kinetics.setStartTemperatureByUser(toTemperature(250.0));
    engine::RingPolymerQMMDEngine rpEngine{settings};
    fillEngineBox(rpEngine);

    setup::RingPolymerSetup setup(rpEngine);
    setup.setupSimulationBox();
    setup.initializeBeads();

    settings::GeneralSettings::setIsRandomSeedSet(false);

    auto &beads = rpEngine.getRingPolymerBeads();

    for (auto &bead : beads)
    {
        bead.calculateDegreesOfFreedom();
        EXPECT_NEAR(temperatureOf(bead), 250.0, 1e-6);

        // velocities are in A/s, so compare with the thermal momentum scale
        double momentumScale = 0.0;
        for (const auto &atom : bead.getAtoms())
        {
            momentumScale += atom->getMass() * norm(atom->getVelocity());
        }

        EXPECT_GT(momentumScale, 0.0);
        EXPECT_LT(norm(bead.calculateMomentum()), 1e-10 * momentumScale);
    }

    // the beads must not all start with the same velocities
    EXPECT_NE(
        beads[0].getAtoms()[0]->getVelocity(),
        beads[1].getAtoms()[0]->getVelocity()
    );
    EXPECT_NE(
        beads[1].getAtoms()[0]->getVelocity(),
        beads[2].getAtoms()[0]->getVelocity()
    );
}

/**
 * @brief the same random seed gives the same velocities of all beads, a
 * different seed gives different ones
 */
TEST_F(TestSetup, ringPolymerSetupVelocitiesAreReproducibleWithRandomSeed)
{
    const auto velocitiesWithSeed = [](const uint_fast32_t seed)
    {
        settings::GeneralSettings::setRandomSeed(seed);
        settings::GeneralSettings::setIsRandomSeedSet(true);

        Settings settings;
        settings.ringPolymerSettings.setNumberOfBeadsByUser(2);
        settings.kinetics.setStartTemperatureByUser(toTemperature(300.0));
        engine::RingPolymerQMMDEngine rpEngine{settings};
        fillEngineBox(rpEngine);

        setup::RingPolymerSetup setup(rpEngine);
        setup.setupSimulationBox();
        setup.initializeBeads();

        settings::GeneralSettings::setIsRandomSeedSet(false);

        auto velocities = std::vector<linalg::Vec3D>();
        for (const auto &bead : rpEngine.getRingPolymerBeads())
        {
            velocities.push_back(bead.getAtoms()[0]->getVelocity());
        }

        return velocities;
    };

    const auto first  = velocitiesWithSeed(5);
    const auto second = velocitiesWithSeed(5);
    const auto other  = velocitiesWithSeed(6);

    EXPECT_EQ(first, second);
    EXPECT_NE(first, other);
    EXPECT_NE(first[0], first[1]);
}

/**
 * @brief with a ring polymer start file the beads are read from it and the
 * velocities are not initialised
 */
TEST_F(TestSetup, ringPolymerSetupReadsBeadsFromRestartFile)
{
    constexpr const char *fileName = "testRingPolymerSetup.rpmd.rst";

    {
        std::ofstream file(fileName);

        for (size_t bead = 1; bead <= 2; ++bead)
        {
            for (size_t index = 1; index <= N_TEST_ATOMS; ++index)
            {
                file << "H" << bead << " " << index << " 1  "
                     << 100.0 * static_cast<double>(bead) << " "
                     << static_cast<double>(index) << " 0.5  0.1 0.2 0.3  "
                     << "1 2 3\n";
            }
        }
    }

    Settings settings;
    settings.ringPolymerSettings.setNumberOfBeadsByUser(2);
    settings.fileSettings.setRingPolymerStartFileNameByUser(fileName);
    engine::RingPolymerQMMDEngine rpEngine{settings};
    fillEngineBox(rpEngine);

    setup::RingPolymerSetup setup(rpEngine);
    setup.setupSimulationBox();
    setup.initializeBeads();

    const auto &beads = rpEngine.getRingPolymerBeads();

    EXPECT_EQ(
        beads[0].getAtoms()[0]->getPosition(),
        linalg::Vec3D(100, 1, 0.5)
    );
    EXPECT_EQ(
        beads[1].getAtoms()[2]->getPosition(),
        linalg::Vec3D(200, 3, 0.5)
    );
    EXPECT_EQ(beads[1].getAtoms()[2]->getVelocity(), linalg::Vec3D(.1, .2, .3));
    EXPECT_EQ(beads[1].getAtoms()[2]->getForce(), linalg::Vec3D(1, 2, 3));

    std::filesystem::remove(fileName);
}

/**
 * @brief setup() runs all three steps in order
 */
TEST_F(
    TestSetup,
    ringPolymerSetupRunsAllStepsAndTheWrapperUsesRingPolymerEngine
)
{
    settings::GeneralSettings::setIsRingPolymerMDActivated(true);

    Settings settings;
    settings.ringPolymerSettings.setNumberOfBeadsByUser(2);
    settings.kinetics.setStartTemperatureByUser(toTemperature(300.0));
    engine::RingPolymerQMMDEngine rpEngine{settings};
    fillEngineBox(rpEngine);

    EXPECT_NO_THROW(setup::setupRingPolymer(rpEngine));
    settings::GeneralSettings::setIsRingPolymerMDActivated(false);

    EXPECT_EQ(rpEngine.getRingPolymerBeads().size(), 2);
    EXPECT_EQ(rpEngine.getRingPolymerBeadsPhysicalData().size(), 2);
    EXPECT_NE(
        rpEngine.getRingPolymerBeads()[0].getAtoms()[0]->getVelocity(),
        linalg::Vec3D(0.0, 0.0, 0.0)
    );
}

/**
 * @brief the wrapper only works with a ring polymer engine
 */
TEST_F(TestSetup, setupRingPolymerWithPlainEngineThrows)
{
    settings::GeneralSettings::setIsRingPolymerMDActivated(true);

    EXPECT_THROW(setup::setupRingPolymer(*_engine), std::bad_cast);

    settings::GeneralSettings::setIsRingPolymerMDActivated(false);
}
