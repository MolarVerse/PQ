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
#include <memory>
#include <string>
#include <vector>

#include "generalSettings.hpp"
#include "outputFileSettings.hpp"
#include "outputFilesSetup.hpp"
#include "ringPolymerqmmdEngine.hpp"
#include "settings.hpp"
#include "simulationBox.hpp"

#ifdef WITH_MPI
#include "mpi.hpp"
#endif

namespace
{
    // settings::OutputFileSettings caches filenames after their first
    // replace from the default sentinel value, so every test in this file
    // reuses the same prefix. On-disk files are wiped before/after each test.
    constexpr const char *PREFIX = "rpmdOutputGateTest";

    void cleanupPrefix()
    {
        const std::vector<std::string> suffixes = {
            ".log",      ".info",     ".rst",        ".en",        ".xyz",
            ".timings",  ".force",    ".instant_en", ".vel",       ".chrg",
            ".mom",      ".vir",      ".stress",     ".box",       ".rpmd.rst",
            ".rpmd.xyz", ".rpmd.vel", ".rpmd.force", ".rpmd.chrg", ".rpmd.en",
            ".ref"
        };

        for (const auto &suffix : suffixes)
            static_cast<void>(
                std::filesystem::remove((std::string(PREFIX) + suffix).c_str())
            );
    }

    [[nodiscard]] bool rpmdRestartFileExists()
    {
        return std::filesystem::exists(std::string(PREFIX) + ".rpmd.rst");
    }

    // setupOutputFiles() already creates the (empty) file via setFilename(),
    // and the beads carry no atoms, so neither existence nor content size
    // can distinguish "written" from "never written". Instead this removes
    // the on-disk file after setup (the already-open ofstream handle stays
    // valid) and relies on RingPolymerRestartFileOutput::write() closing and
    // re-opening the file on every call: if writeOutput() is gated away for
    // this rank, the file simply never comes back.
    void removeRpmdRestartFile()
    {
        static_cast<void>(
            std::filesystem::remove(std::string(PREFIX) + ".rpmd.rst")
        );
    }
}   // namespace

/**
 * @class TestRingPolymerEngineOutputGating
 *
 * @brief Fixture building a minimal, directly-usable RingPolymerQMMDEngine
 * (no QM runner invoked) with exactly two trivial beads, so
 * RingPolymerEngine::writeOutput can be exercised without a full simulation
 * setup.
 */
class TestRingPolymerEngineOutputGating : public ::testing::Test
{
   protected:
    std::unique_ptr<Settings>                      _settings;
    std::unique_ptr<engine::RingPolymerQMMDEngine> _engine;

    void SetUp() override
    {
        cleanupPrefix();

        settings::GeneralSettings::setJobtype(JobType::RING_POLYMER_QM_MD);
        settings::GeneralSettings::setIsRingPolymerMDActivated(true);

        _settings = std::make_unique<Settings>();
        _settings->outputFiles.setFilePrefix(PREFIX);
        _settings->outputFiles.setOutputFrequency(1);
        _settings->finalize();

        _engine = std::make_unique<engine::RingPolymerQMMDEngine>(*_settings);

        _engine->addRingPolymerBead(molsys::SimulationBox());
        _engine->addRingPolymerBead(molsys::SimulationBox());
        _engine->resizeRingPolymerBeadPhysicalData(2);

        setup::setupOutputFiles(*_engine);
    }

    void TearDown() override
    {
#ifdef WITH_MPI
        mpi::MPI::setRank(0);
#endif
        _engine->getXyzOutput().close();
        _engine->getVelOutput().close();
        _engine->getForceOutput().close();
        _engine->getChargeOutput().close();
        _engine->getRstFileOutput().close();
        _engine->getRingPolymerRstFileOutput().close();
        _engine->getRingPolymerXyzOutput().close();
        _engine->getRingPolymerVelOutput().close();
        _engine->getRingPolymerForceOutput().close();
        _engine->getRingPolymerChargeOutput().close();
        _engine->getEnergyOutput().close();
        _engine->getInstantEnergyOutput().close();
        _engine->getMomentumOutput().close();
        _engine->getRingPolymerEnergyOutput().close();
        _engine->getTimingsOutput().close();
        _engine->getLogOutput().close();
        _engine->getInfoOutput().close();

        settings::GeneralSettings::setIsRingPolymerMDActivated(false);

        cleanupPrefix();
    }
};

/**
 * @brief On the root rank (the default/only rank outside an mpiexec launch),
 * writeOutput must still write the ring polymer restart file.
 */
TEST_F(TestRingPolymerEngineOutputGating, rootRankWritesRingPolymerRstFile)
{
    removeRpmdRestartFile();
    ASSERT_FALSE(rpmdRestartFileExists());

    _engine->writeOutput(_settings->outputFiles);

    EXPECT_TRUE(rpmdRestartFileExists());
}

#ifdef WITH_MPI
/**
 * @brief A non-root rank must not write the ring polymer restart file - each
 * MPI rank already holds the full, synchronized bead data (see
 * RingPolymerQMMDEngine::qmCalculation/applyThermostatHalfStep, which gather
 * via MPI_Bcast), so letting every rank write duplicates the write and can
 * race/fail depending on filesystem timing (the root cause of the
 * regression_mpi_qm_rpmd_smoke_zip CI flake this test guards against).
 */
TEST_F(
    TestRingPolymerEngineOutputGating,
    nonRootRankDoesNotWriteRingPolymerRstFile
)
{
    removeRpmdRestartFile();
    ASSERT_FALSE(rpmdRestartFileExists());

    mpi::MPI::setRank(1);
    _engine->writeOutput(_settings->outputFiles);
    mpi::MPI::setRank(0);

    EXPECT_FALSE(rpmdRestartFileExists());
}
#endif
