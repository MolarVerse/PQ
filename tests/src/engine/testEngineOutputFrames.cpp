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
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>

#include "atom.hpp"
#include "engineOutput.hpp"
#include "enums/rstMetaData.hpp"
#include "molecule.hpp"
#include "orthorhombicBox.hpp"
#include "outputFileSettings.hpp"
#include "referenceSettings.hpp"
#include "simulationBox.hpp"
#include "thermostat.hpp"
#include "units.hpp"
#include "vector3d.hpp"
#include "version.hpp"

namespace
{
    molsys::SimulationBox makeSimulationBox()
    {
        molsys::SimulationBox simulationBox;
        simulationBox.setBox(
            molsys::OrthorhombicBox{
                {10.0 * angstrom, 10.0 * angstrom, 10.0 * angstrom}
            },
            false
        );

        molsys::Molecule molecule{MolType{1}};

        for (std::size_t i = 0; i < 2; ++i)
        {
            auto       atom = std::make_shared<molsys::Atom>();
            const auto x    = static_cast<double>(i) + 1.0;
            atom->setName(i == 0 ? "O" : "H");
            atom->setPosition(
                {x * angstrom, 2.0 * x * angstrom, 3.0 * x * angstrom}
            );
            atom->setVelocity(linalg::Vec3D{0.1 * x, 0.0, 0.0} * angstrom / s);
            atom->setForce(
                linalg::Vec3D{0.0, -x, 0.0} * kcal_per_mol / angstrom
            );
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

    // writes 3 frames with async output on or off, as xyz files or one
    // extxyz file; returns the files' content
    std::string writeFrames(bool async, bool extxyz)
    {
        const std::string prefix =
            std::string(async ? "engineOutputAsync" : "engineOutputSync") +
            (extxyz ? "Ext" : "");

        OutputFileSettings outputFileSettings;
        outputFileSettings.setAsyncOutputByUser(async);
        const ReferenceSettings referenceSettings;

        std::string content;

        {
            engine::EngineOutput output{outputFileSettings, referenceSettings};
            output.getRstFileOutput().setFilename(prefix + ".rst", true);

            if (extxyz)
            {
                output.openExtxyzOutput(prefix + ".extxyz", true);
            }
            else
            {
                output.getXyzOutput().setFilename(prefix + ".xyz", true);
                output.getVelOutput().setFilename(prefix + ".vel", true);
                output.getForceOutput().setFilename(prefix + ".force", true);
                output.getChargeOutput().setFilename(prefix + ".chrg", true);
            }

            const auto                   simulationBox = makeSimulationBox();
            const thermostat::Thermostat thermostat{};

            for (std::size_t step = 1; step <= 3; ++step)
            {
                output.writeFrame(
                    simulationBox,
                    thermostat,
                    step,
                    -1.5 * static_cast<double>(step) * kcal_per_mol,
                    100 * fs
                );
            }

            output.waitForFrames();
        }

        for (const auto *suffix :
             {".xyz", ".vel", ".force", ".chrg", ".rst", ".extxyz"})
        {
            const auto name  = prefix + suffix;
            content         += std::string(suffix) + ":" + readFile(name);
            std::filesystem::remove(name);
        }

        return content;
    }
}   // namespace

TEST(TestEngineOutputFrames, asyncAndSyncWriteTheSameFiles)
{
    for (const bool extxyz : {false, true})
    {
        const auto sync = writeFrames(false, extxyz);

        EXPECT_FALSE(sync.empty());
        EXPECT_EQ(writeFrames(true, extxyz), sync);
    }
}

TEST(TestEngineOutputFrames, extxyzReplacesTheXyzFiles)
{
    const auto content = writeFrames(false, true);

    // the empty xyz, vel, force and chrg files are followed by the restart
    // file, which starts with its metadata line
    EXPECT_NE(
        content.find(
            std::string(".xyz:.vel:.force:.chrg:.rst:") +
            RestartFileMetaDataKeyWord + " " +
            RestartFileMetaDataMeta::toString(RestartFileMetaData::Version) +
            " " + CURRENT_VERSION.to_string() + "\nStep"
        ),
        std::string::npos
    );
    EXPECT_NE(content.find("units=pq"), std::string::npos);
}

TEST(TestEngineOutputFrames, aseTrajReplacesTheXyzFiles)
{
    const std::string prefix = "engineOutputAse";

    OutputFileSettings outputFileSettings;
    outputFileSettings.setAsyncOutputByUser(false);
    const ReferenceSettings referenceSettings;

    {
        engine::EngineOutput output{outputFileSettings, referenceSettings};
        output.getRstFileOutput().setFilename(prefix + ".rst", true);
        output.openAseTrajOutput(prefix + ".ase.traj", true);

        const auto                   simulationBox = makeSimulationBox();
        const thermostat::Thermostat thermostat{};

        for (std::size_t step = 1; step <= 3; ++step)
        {
            output.writeFrame(
                simulationBox,
                thermostat,
                step,
                -1.5 * kcal_per_mol,
                100 * fs
            );
        }

        output.waitForFrames();
    }

    const auto   content = readFile(prefix + ".ase.traj");
    std::int64_t nItems  = 0;
    std::memcpy(&nItems, content.substr(32, 8).data(), sizeof(nItems));

    EXPECT_EQ(nItems, 3);
    EXPECT_NE(content.find(R"("info": {"step": 3}})"), std::string::npos);

    for (const auto *suffix : {".xyz", ".vel", ".force", ".chrg"})
    {
        EXPECT_FALSE(std::filesystem::exists(prefix + suffix)) << suffix;
    }

    std::filesystem::remove(prefix + ".ase.traj");
    std::filesystem::remove(prefix + ".rst");
}

TEST(TestEngineOutputFrames, asyncOutputFitsInHalfOfTheMemory)
{
    constexpr std::size_t nAtoms = 1'000'000;
    constexpr std::size_t GiB    = std::size_t{1} << 30U;

    EXPECT_TRUE(engine::EngineOutput::asyncOutputFits(nAtoms, std::nullopt));
    EXPECT_TRUE(engine::EngineOutput::asyncOutputFits(nAtoms, 64 * GiB));
    EXPECT_FALSE(engine::EngineOutput::asyncOutputFits(nAtoms, GiB / 4));
}
