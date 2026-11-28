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
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "atom.hpp"
#include "exceptions.hpp"
#include "frame.hpp"
#include "molecule.hpp"
#include "orthorhombicBox.hpp"
#include "simulationBox.hpp"
#include "throwWithMessage.hpp"
#include "units.hpp"
#include "vector3d.hpp"
#include "version.hpp"
#include "zarrOutput.hpp"
#include "zarrStore.hpp"

namespace
{
    struct AtomData
    {
        const char   *name;
        linalg::Vec3D position;
        linalg::Vec3D velocity;
        linalg::Vec3D force;
        double        charge;
    };

    const std::vector<AtomData> ATOMS = {
        {.name     = "O",
         .position = {1.0, 2.0, 3.0},
         .velocity = {0.01, 0.02, 0.03},
         .force    = {1.0, 2.0, 3.0},
         .charge   = -0.8},
        {.name     = "H",
         .position = {12.0, 2.0, 3.0},
         .velocity = {-0.01, 0.0, 0.0},
         .force    = {-1.0, 0.0, 0.0},
         .charge   = 0.4},
        {.name     = "dum",
         .position = {1.0, 1.0, 1.0},
         .velocity = {0.0, 0.0, 0.0},
         .force    = {0.0, 0.0, 0.0},
         .charge   = 0.4},
    };

    out::Frame makeFrame(std::size_t nAtoms)
    {
        molsys::SimulationBox simulationBox;
        simulationBox.setBox(
            molsys::OrthorhombicBox{
                {10.0 * angstrom, 11.0 * angstrom, 12.0 * angstrom}
            },
            false
        );

        molsys::Molecule molecule{MolType{1}};
        for (std::size_t index = 0; index < nAtoms; ++index)
        {
            const auto &data = ATOMS[index];
            auto        atom = std::make_shared<molsys::Atom>();
            atom->setName(data.name);
            atom->setPosition(data.position * angstrom);
            atom->setVelocity(data.velocity * angstrom / fs);
            atom->setForce(data.force * kcal_per_mol / angstrom);
            atom->setPartialCharge(data.charge * elementary_charge);
            molecule.addAtom(atom);
            simulationBox.addAtom(atom);
        }
        simulationBox.addMolecule(molecule);

        auto frame            = out::Frame{simulationBox};
        frame.potentialEnergy = -1.5 * kcal_per_mol;
        return frame;
    }

    std::string metadata(const std::filesystem::path &path)
    {
        return zarrTest::readFile(path / "zarr.json");
    }

    bool contains(const std::string &text, const std::string &part)
    {
        return text.find(part) != std::string::npos;
    }
}   // namespace

/**
 * @brief the H5MD groups, arrays and units, values as float32 in PQ units
 *
 */
TEST(TestZarrOutput, writesH5mdLayoutInPqUnits)
{
    const std::filesystem::path path  = "testZarrOutput.traj.zarr";
    const auto                  codec = zarrTest::outputCodec();
    std::filesystem::remove_all(path);

    {
        out::ZarrOutput output{path.string(), false, 0.5 * fs};
        output.write(makeFrame(3), 10);
        output.write(makeFrame(3), 11);
        output.close();
    }

    EXPECT_EQ(
        metadata(path),
        R"({"zarr_format": 3, "node_type": "group", "attributes": {"h5md": {"version": [1, 1], "author": {"name": "PQ"}, "creator": {"name": "PQ", "version": ")" +
            CURRENT_VERSION.to_string() + R"("}}}})"
    );

    const auto all = path / "particles" / "all";
    EXPECT_TRUE(contains(metadata(path / "particles"), R"("node_type": "group")"));
    EXPECT_EQ(
        zarrTest::readChunk<std::int32_t>(all / "species" / "c" / "0", codec),
        (std::vector<std::int32_t>{8, 1, 0})
    );
    EXPECT_TRUE(contains(
        metadata(all / "box"),
        R"("attributes": {"dimension": 3, "boundary": ["periodic", "periodic", "periodic"]})"
    ));
    EXPECT_TRUE(contains(
        metadata(path / "parameters"),
        R"("attributes": {"atom_names": ["O", "H", "dum"]})"
    ));

    const auto position = metadata(all / "position" / "value");
    EXPECT_TRUE(contains(position, R"("shape": [2, 3, 3], "data_type": "float32")"));
    EXPECT_TRUE(contains(position, R"("chunk_shape": [100, 3, 3])"));
    EXPECT_TRUE(contains(position, R"("attributes": {"unit": "Angstrom"})"));
    EXPECT_TRUE(contains(
        metadata(all / "velocity" / "value"),
        R"("unit": "Angstrom fs-1")"
    ));
    EXPECT_TRUE(contains(
        metadata(all / "force" / "value"),
        R"("unit": "kcal mol-1 Angstrom-1")"
    ));
    EXPECT_TRUE(contains(metadata(all / "charge" / "value"), R"("shape": [2, 3])"));
    EXPECT_TRUE(contains(
        metadata(all / "box" / "edges" / "value"),
        R"("shape": [2, 3, 3], "data_type": "float64")"
    ));
    const auto energy = path / "observables" / "potential_energy";
    EXPECT_TRUE(contains(
        metadata(energy / "value"),
        R"("shape": [2], "data_type": "float64")"
    ));

    const auto frameValues = [&](const std::filesystem::path &value)
    {
        auto values = zarrTest::readChunk<float>(value / "c" / "0" / "0" / "0", codec);
        values.resize(2 * 9);
        return values;
    };

    // the H atom is wrapped into the box
    const auto positions  = frameValues(all / "position" / "value");
    const auto velocities = frameValues(all / "velocity" / "value");
    const auto forces     = frameValues(all / "force" / "value");
    for (std::size_t atomIndex = 0; atomIndex < 3; ++atomIndex)
    {
        auto expected = ATOMS[atomIndex].position;
        if (expected[0] > 5.0)
        {
            expected[0] -= 10.0;
        }

        for (std::size_t dim = 0; dim < 3; ++dim)
        {
            const auto index = (3 * atomIndex) + dim;
            EXPECT_FLOAT_EQ(positions[index], static_cast<float>(expected[dim]));
            EXPECT_FLOAT_EQ(positions[index + 9], positions[index]);
            EXPECT_FLOAT_EQ(
                velocities[index],
                static_cast<float>(ATOMS[atomIndex].velocity[dim])
            );
            EXPECT_FLOAT_EQ(
                forces[index],
                static_cast<float>(ATOMS[atomIndex].force[dim])
            );
        }
    }

    auto charges =
        zarrTest::readChunk<float>(all / "charge" / "value" / "c" / "0" / "0", codec);
    charges.resize(3);
    EXPECT_EQ(charges, (std::vector<float>{-0.8F, 0.4F, 0.4F}));

    auto edges = zarrTest::readChunk<double>(
        all / "box" / "edges" / "value" / "c" / "0" / "0" / "0",
        codec
    );
    edges.resize(9);
    EXPECT_EQ(
        edges,
        (std::vector<double>{10.0, 0.0, 0.0, 0.0, 11.0, 0.0, 0.0, 0.0, 12.0})
    );

    auto steps =
        zarrTest::readChunk<std::int64_t>(energy / "step" / "c" / "0", codec);
    auto times = zarrTest::readChunk<double>(energy / "time" / "c" / "0", codec);
    auto energies =
        zarrTest::readChunk<double>(energy / "value" / "c" / "0", codec);
    steps.resize(2);
    times.resize(2);
    energies.resize(2);
    EXPECT_EQ(steps, (std::vector<std::int64_t>{10, 11}));
    EXPECT_EQ(times, (std::vector<double>{5.0, 5.5}));
    EXPECT_EQ(energies, (std::vector<double>{-1.5, -1.5}));

    std::filesystem::remove_all(path);
}

/**
 * @brief full chunks are on disk before close; close adds the rest
 *
 */
TEST(TestZarrOutput, writesFullChunksBeforeClose)
{
    const std::filesystem::path path = "testZarrOutputChunks.traj.zarr";
    const auto value = path / "observables" / "potential_energy" / "value";
    std::filesystem::remove_all(path);

    out::ZarrOutput output{path.string(), true, 1.0 * fs};
    output.close();
    EXPECT_FALSE(std::filesystem::exists(path / "observables"));

    for (std::size_t step = 0; step <= out::ZARR_CHUNK_FRAMES; ++step)
    {
        output.write(makeFrame(2), step);
    }

    EXPECT_TRUE(contains(metadata(value), R"("shape": [100])"));
    EXPECT_TRUE(std::filesystem::exists(value / "c" / "0"));
    EXPECT_FALSE(std::filesystem::exists(value / "c" / "1"));

    output.close();
    EXPECT_TRUE(contains(metadata(value), R"("shape": [101])"));
    EXPECT_TRUE(std::filesystem::exists(value / "c" / "1"));

    std::filesystem::remove_all(path);
}

/**
 * @brief errors are reported, not silently ignored
 *
 */
TEST(TestZarrOutput, throwsOnInvalidUse)
{
    const std::filesystem::path path     = "testZarrOutputErrors.traj.zarr";
    const auto                  filename = path.string();
    std::filesystem::remove_all(path);

    {
        out::ZarrOutput output{filename, false, 1.0 * fs};
        output.write(makeFrame(3), 0);

        EXPECT_THROW_MSG(
            output.write(makeFrame(2), 1),
            exc::InputFileException,
            "Zarr output needs the same number of atoms in every frame"
        );
    }

    EXPECT_THROW_MSG(
        out::ZarrOutput(filename, false, 1.0 * fs),
        exc::InputFileException,
        "File already exists - filename = " + filename
    );

    // an existing store is replaced
    {
        out::ZarrOutput output{filename, true, 1.0 * fs};
    }
    EXPECT_TRUE(std::filesystem::exists(path / "zarr.json"));
    EXPECT_FALSE(std::filesystem::exists(path / "particles"));

    // a directory that is not a store is kept
    std::filesystem::remove_all(path);
    std::filesystem::create_directories(path);
    std::ofstream{path / "keep.txt"} << "data";

    EXPECT_THROW_MSG(
        out::ZarrOutput(filename, true, 1.0 * fs),
        exc::InputFileException,
        "Directory is not a Zarr store and is not replaced - filename = " +
            filename
    );
    EXPECT_TRUE(std::filesystem::exists(path / "keep.txt"));

    std::filesystem::remove_all(path);
}
