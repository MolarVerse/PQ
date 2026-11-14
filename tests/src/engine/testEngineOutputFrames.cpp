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

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>

#include "atom.hpp"
#include "engineOutput.hpp"
#include "molecule.hpp"
#include "orthorhombicBox.hpp"
#include "outputFileSettings.hpp"
#include "referenceSettings.hpp"
#include "simulationBox.hpp"
#include "thermostat.hpp"

namespace
{
    molsys::SimulationBox makeSimulationBox()
    {
        molsys::SimulationBox simulationBox;
        simulationBox.setBox(
            molsys::OrthorhombicBox{{10.0, 10.0, 10.0}},
            false
        );

        molsys::Molecule molecule{MolType{1}};

        for (std::size_t i = 0; i < 2; ++i)
        {
            auto       atom = std::make_shared<molsys::Atom>();
            const auto x    = static_cast<double>(i) + 1.0;
            atom->setName(i == 0 ? "O" : "H");
            atom->setPosition({x, 2.0 * x, 3.0 * x});
            atom->setVelocity({0.1 * x, 0.0, 0.0});
            atom->setForce({0.0, -x, 0.0});
            molecule.addAtom(atom);
            simulationBox.addAtom(atom);
        }

        simulationBox.addMolecule(molecule);

        return simulationBox;
    }

    std::string readFile(const std::string &name)
    {
        std::ifstream      file(name);
        std::ostringstream content;
        content << file.rdbuf();
        return content.str();
    }

    // writes 3 frames with async output on or off; returns the files' content
    std::string writeFrames(const bool async)
    {
        const std::string prefix =
            async ? "engineOutputAsync" : "engineOutputSync";

        OutputFileSettings outputFileSettings;
        outputFileSettings.setAsyncOutputByUser(async);
        const ReferenceSettings referenceSettings;

        std::string content;

        {
            engine::EngineOutput output{outputFileSettings, referenceSettings};
            output.getXyzOutput().setFilename(prefix + ".xyz", true);
            output.getVelOutput().setFilename(prefix + ".vel", true);
            output.getForceOutput().setFilename(prefix + ".force", true);
            output.getChargeOutput().setFilename(prefix + ".chrg", true);
            output.getRstFileOutput().setFilename(prefix + ".rst", true);

            const auto                   simulationBox = makeSimulationBox();
            const thermostat::Thermostat thermostat{};

            for (std::size_t step = 1; step <= 3; ++step)
                output.writeFrame(simulationBox, thermostat, step);

            output.waitForFrames();
        }

        for (const auto *suffix : {".xyz", ".vel", ".force", ".chrg", ".rst"})
        {
            content += readFile(prefix + suffix);
            std::filesystem::remove(prefix + suffix);
        }

        return content;
    }
}   // namespace

TEST(TestEngineOutputFrames, asyncAndSyncWriteTheSameFiles)
{
    const auto sync = writeFrames(false);

    EXPECT_FALSE(sync.empty());
    EXPECT_EQ(writeFrames(true), sync);
}

TEST(TestEngineOutputFrames, asyncOutputFitsInHalfOfTheMemory)
{
    constexpr std::size_t nAtoms = 1'000'000;
    constexpr std::size_t GiB    = std::size_t{1} << 30U;

    EXPECT_TRUE(engine::EngineOutput::asyncOutputFits(nAtoms, std::nullopt));
    EXPECT_TRUE(engine::EngineOutput::asyncOutputFits(nAtoms, 64 * GiB));
    EXPECT_FALSE(engine::EngineOutput::asyncOutputFits(nAtoms, GiB / 4));
}
