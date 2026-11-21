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

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

#include "atom.hpp"
#include "exceptions.hpp"
#include "generalSettings.hpp"
#include "molecule.hpp"
#include "ringPolymerRestartFileOutput.hpp"
#include "ringPolymerRestartFileReader.hpp"
#include "ringPolymerSettings.hpp"
#include "ringPolymerqmmdEngine.hpp"
#include "settings.hpp"
#include "simulationBox.hpp"
#include "throwWithMessage.hpp"

namespace
{
    constexpr const char *FILE_NAME =
        "testRingPolymerRestartFileReader.rpmd.rst";

    molsys::SimulationBox makeBead(const std::string &namePrefix)
    {
        auto box      = molsys::SimulationBox();
        auto molecule = molsys::Molecule();

        for (size_t i = 0; i < 2; ++i)
        {
            auto atom = std::make_shared<molsys::Atom>();
            atom->setName(namePrefix);
            molecule.addAtom(atom);
            box.addAtom(atom);
        }

        box.addMolecule(molecule);

        return box;
    }

    void writeFile(const std::string &content)
    {
        std::ofstream file(FILE_NAME);
        file << content;
    }
}   // namespace

/**
 * @class TestRingPolymerRestartFileReader
 *
 * @brief fixture with a ring polymer engine of two beads with two atoms each
 */
class TestRingPolymerRestartFileReader : public ::testing::Test
{
   protected:
    std::unique_ptr<Settings>                      _settings;
    std::unique_ptr<engine::RingPolymerQMMDEngine> _engine;

    void SetUp() override
    {
        _settings = std::make_unique<Settings>();
        _settings->generalSettings.setJobtypeByUser(
            JobType::RING_POLYMER_QM_MD
        );
        _settings->ringPolymerSettings.setNumberOfBeadsByUser(2);
        _settings->fileSettings.setRingPolymerStartFileNameByUser(FILE_NAME);
        _settings->timings.setNumberOfStepsByUser(1);

        _settings->finalize();

        _engine = std::make_unique<engine::RingPolymerQMMDEngine>(*_settings);
        _engine->addRingPolymerBead(makeBead("H"));
        _engine->addRingPolymerBead(makeBead("O"));
    }

    void TearDown() override { std::filesystem::remove(FILE_NAME); }
};

/**
 * @brief positions, velocities and forces of every atom of every bead are read
 * from the 12-element lines, in the order bead by bead
 */
TEST_F(TestRingPolymerRestartFileReader, readsPositionsVelocitiesAndForces)
{
    writeFile(
        "H1 1 1  1.0 2.0 3.0  0.1 0.2 0.3  10.0 20.0 30.0\n"
        "H1 2 1  4.0 5.0 6.0  0.4 0.5 0.6  40.0 50.0 60.0\n"
        "O2 1 1  7.0 8.0 9.0  0.7 0.8 0.9  70.0 80.0 90.0\n"
        "O2 2 1  1.5 2.5 3.5  1.1 1.2 1.3  11.0 12.0 13.0\n"
    );

    input::ringPolymer::RingPolymerRestartFileReader reader(
        FILE_NAME,
        *_engine
    );
    ASSERT_NO_THROW(reader.read(2));

    const auto &beads = _engine->getRingPolymerBeads();

    EXPECT_EQ(beads[0].getAtoms()[0]->getPosition(), linalg::Vec3D(1, 2, 3));
    EXPECT_EQ(beads[0].getAtoms()[1]->getVelocity(), linalg::Vec3D(.4, .5, .6));
    EXPECT_EQ(beads[0].getAtoms()[1]->getForce(), linalg::Vec3D(40, 50, 60));
    EXPECT_EQ(beads[1].getAtoms()[0]->getPosition(), linalg::Vec3D(7, 8, 9));
    EXPECT_EQ(beads[1].getAtoms()[1]->getForce(), linalg::Vec3D(11, 12, 13));
}

/**
 * @brief comments and empty lines between the atom lines are skipped, and a
 * line with 21 elements (with the extra columns of the non-restart output) is
 * accepted
 */
TEST_F(TestRingPolymerRestartFileReader, skipsCommentsAndEmptyLinesAndAccepts21)
{
    writeFile(
        "# a comment\n"
        "\n"
        "H1 1 1  1 2 3  .1 .2 .3  4 5 6  # trailing comment\n"
        "   \n"
        "H1 2 1  1 2 3  .1 .2 .3  4 5 6  0 0 0  0 0 0  0 0 0\n"
        "O2 1 1  1 2 3  .1 .2 .3  4 5 6\n"
        "O2 2 1  1 2 3  .1 .2 .3  4 5 6\n"
    );

    input::ringPolymer::RingPolymerRestartFileReader reader(
        FILE_NAME,
        *_engine
    );

    EXPECT_NO_THROW(reader.read(2));
    EXPECT_EQ(
        _engine->getRingPolymerBeads()[0].getAtoms()[1]->getForce(),
        linalg::Vec3D(4, 5, 6)
    );
}

/**
 * @brief a file with fewer atom lines than the beads need is an error
 */
TEST_F(TestRingPolymerRestartFileReader, tooFewLinesThrow)
{
    writeFile(
        "H1 1 1  1 2 3  .1 .2 .3  4 5 6\n"
        "H1 2 1  1 2 3  .1 .2 .3  4 5 6\n"
        "O2 1 1  1 2 3  .1 .2 .3  4 5 6\n"
    );

    input::ringPolymer::RingPolymerRestartFileReader reader(
        FILE_NAME,
        *_engine
    );

    ASSERT_THROW_MSG(
        reader.read(2),
        exc::RingPolymerRestartFileException,
        "Error reading ring polymer restart file"
    );
}

/**
 * @brief a line with a number of elements other than 12 or 21 names its line
 */
TEST_F(TestRingPolymerRestartFileReader, wrongNumberOfElementsThrows)
{
    writeFile(
        "H1 1 1  1 2 3  .1 .2 .3  4 5 6\n"
        "H1 2 1  1 2 3  .1 .2 .3  4 5\n"
    );

    input::ringPolymer::RingPolymerRestartFileReader reader(
        FILE_NAME,
        *_engine
    );

    ASSERT_THROW_MSG(
        reader.read(2),
        exc::RingPolymerRestartFileException,
        "Error in line 2: Atom section must have 12 or 21 elements"
    );
}

/**
 * @brief the wrapper takes the file name and the number of beads from the
 * settings
 */
TEST_F(TestRingPolymerRestartFileReader, wrapperUsesSettings)
{
    writeFile(
        "H1 1 1  1 2 3  .1 .2 .3  4 5 6\n"
        "H1 2 1  1 2 3  .1 .2 .3  4 5 6\n"
        "O2 1 1  7 8 9  .1 .2 .3  4 5 6\n"
        "O2 2 1  1 2 3  .1 .2 .3  4 5 6\n"
    );

    ASSERT_NO_THROW(input::ringPolymer::readRingPolymerRestartFile(*_engine));
    EXPECT_EQ(
        _engine->getRingPolymerBeads()[1].getAtoms()[0]->getPosition(),
        linalg::Vec3D(7, 8, 9)
    );
}

/**
 * @brief what the writer writes, the reader reads back
 */
TEST_F(TestRingPolymerRestartFileReader, readsBackWhatTheWriterWrote)
{
    auto &beads = _engine->getRingPolymerBeads();

    double value = 0.25;
    for (auto &bead : beads)
    {
        for (auto &atom : bead.getAtoms())
        {
            atom->setPosition({value, value + 1.0, value + 2.0});
            atom->setVelocity({value * 1e-3, value * 2e-3, value * 3e-3});
            atom->setForce({-value, value * 2.0, -value * 3.0});
            value += 0.5;
        }
    }

    out::RingPolymerRestartFileOutput writer(FILE_NAME);
    writer.setFilename(FILE_NAME, true);
    writer.write(beads);
    writer.close();

    auto expected = std::vector<linalg::Vec3D>();
    for (const auto &bead : beads)
    {
        for (const auto &atom : bead.getAtoms())
        {
            expected.push_back(atom->getPosition());
        }
    }

    for (auto &bead : beads)
    {
        for (auto &atom : bead.getAtoms())
        {
            atom->setPosition({0, 0, 0});
            atom->setVelocity({0, 0, 0});
            atom->setForce({0, 0, 0});
        }
    }

    input::ringPolymer::RingPolymerRestartFileReader reader(
        FILE_NAME,
        *_engine
    );
    ASSERT_NO_THROW(reader.read(2));

    size_t atomIndex = 0;
    for (const auto &bead : beads)
    {
        for (const auto &atom : bead.getAtoms())
        {
            for (size_t axis = 0; axis < 3; ++axis)
            {
                EXPECT_NEAR(
                    atom->getPosition()[axis],
                    expected[atomIndex][axis],
                    1e-7
                );
            }

            EXPECT_NE(atom->getVelocity(), linalg::Vec3D(0, 0, 0));
            EXPECT_NE(atom->getForce(), linalg::Vec3D(0, 0, 0));
            ++atomIndex;
        }
    }
}
