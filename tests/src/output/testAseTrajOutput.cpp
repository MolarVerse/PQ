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
#include <vector>

#include "aseTrajOutput.hpp"
#include "atom.hpp"
#include "exceptions.hpp"
#include "frame.hpp"
#include "molecule.hpp"
#include "orthorhombicBox.hpp"
#include "simulationBox.hpp"
#include "throwWithMessage.hpp"
#include "ulmWriter.hpp"
#include "units.hpp"
#include "vector3d.hpp"

namespace
{
    // ase.units (CODATA 2014) as read by ASE 3.29
    constexpr double ASE_FS           = 0.09822694788464063;
    constexpr double ASE_KCAL_PER_MOL = 0.04336410390059322;
    constexpr double TOLERANCE        = 1e-6;

    struct AtomData
    {
        const char   *name;
        double        mass;
        linalg::Vec3D position;
        linalg::Vec3D velocity;
        linalg::Vec3D force;
        double        charge;
    };

    const std::vector<AtomData> ATOMS = {
        {.name     = "O",
         .mass     = 15.999,
         .position = {1.0, 2.0, 3.0},
         .velocity = {0.01, 0.02, 0.03},
         .force    = {1.0, 2.0, 3.0},
         .charge   = -0.8},
        {.name     = "H",
         .mass     = 1.008,
         .position = {2.0, 2.0, 3.0},
         .velocity = {-0.01, 0.0, 0.0},
         .force    = {-1.0, 0.0, 0.0},
         .charge   = 0.4},
        {.name     = "dum",
         .mass     = 0.0,
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
            atom->setMass(data.mass * g_per_mol);
            atom->setPosition(data.position * angstrom);
            atom->setVelocity(data.velocity * angstrom / fs);
            atom->setForce(data.force * kcal_per_mol / angstrom);
            atom->setPartialCharge(data.charge * elementary_charge);
            molecule.addAtom(atom);
            simulationBox.addAtom(atom);
        }
        simulationBox.addMolecule(molecule);

        auto frame            = out::Frame{simulationBox};
        frame.potentialEnergy = -1.5;
        return frame;
    }

    std::string readFile(const std::string &name)
    {
        std::ifstream      file(name, std::ios::binary);
        std::ostringstream content;
        content << file.rdbuf();
        return content.str();
    }

    template <typename T>
    T readValue(const std::string &data, std::int64_t position)
    {
        T          value{};
        const auto bytes =
            data.substr(static_cast<std::size_t>(position), sizeof(T));
        std::memcpy(&value, bytes.data(), sizeof(T));
        return value;
    }

    // JSON text of item index
    std::string itemJson(const std::string &data, std::int64_t index)
    {
        const auto offsets =
            readValue<std::int64_t>(data, out::ulm::OFFSETS_POINTER_POSITION);
        const auto position = readValue<std::int64_t>(
            data,
            offsets + (index * static_cast<std::int64_t>(sizeof(std::int64_t)))
        );
        const auto length = readValue<std::int64_t>(data, position);

        return data.substr(
            static_cast<std::size_t>(position) + sizeof(std::int64_t),
            static_cast<std::size_t>(length)
        );
    }

    // byte offset of array name in the JSON text
    std::int64_t arrayOffset(const std::string &json, const std::string &name)
    {
        const auto key = json.find('"' + name + R"(.": {"ndarray")");
        EXPECT_NE(key, std::string::npos) << name;

        const auto end   = json.find("]}", key);
        const auto start = json.rfind(' ', end) + 1;
        return std::stoll(json.substr(start, end - start));
    }

    template <typename T>
    std::vector<T> readArray(
        const std::string &data,
        const std::string &json,
        const std::string &name,
        std::size_t        size
    )
    {
        const auto     offset = arrayOffset(json, name);
        std::vector<T> values(size);

        EXPECT_EQ(offset % 8, 0) << name;
        const auto bytes =
            data.substr(static_cast<std::size_t>(offset), size * sizeof(T));
        std::memcpy(values.data(), bytes.data(), bytes.size());
        return values;
    }

    double jsonValue(const std::string &json, const std::string &name)
    {
        const auto key = json.find('"' + name + R"(": )");
        EXPECT_NE(key, std::string::npos) << name;

        return std::stod(json.substr(key + name.size() + 4));
    }
}   // namespace

/**
 * @brief header, offsets, JSON and arrays as ASE writes them, in ASE units
 *
 */
TEST(TestAseTrajOutput, writesUlmFramesInAseUnits)
{
    const std::string filename = "testAseTrajOutput.traj";

    {
        out::AseTrajOutput output{filename};
        output.setFilename(filename, true);
        output.write(makeFrame(3), 7);
        output.write(makeFrame(3), 8);
        output.close();
    }

    const auto data = readFile(filename);

    EXPECT_EQ(data.substr(0, 24), "- of UlmASE-Trajectory  ");
    EXPECT_EQ(readValue<std::int64_t>(data, 24), 3);
    EXPECT_EQ(readValue<std::int64_t>(data, out::ulm::ITEM_COUNT_POSITION), 2);

    // the second frame moved the offsets table to the end of the file
    const auto offsets =
        readValue<std::int64_t>(data, out::ulm::OFFSETS_POINTER_POSITION);
    EXPECT_GT(offsets, out::ulm::HEADER_SIZE);
    EXPECT_EQ(offsets % 8, 0);
    EXPECT_EQ(
        readValue<std::int64_t>(data, offsets),
        readValue<std::int64_t>(data, out::ulm::HEADER_SIZE)
    );

    const auto json = itemJson(data, 0);
    EXPECT_TRUE(json.starts_with(
        R"({"version": 1, "ase_version": "3.29.0", "pbc": [true, true, true], "numbers.": {"ndarray": [[3], "int64", 56]}, "masses.": {"ndarray": [[3], "float64", 80]}, "positions.": {"ndarray": [[3, 3], "float64", 104]}, "cell": [[1e+01, 0e+00, 0e+00], [0e+00, 1.1e+01, 0e+00], [0e+00, 0e+00, 1.2e+01]], )"
    )) << json;
    EXPECT_NE(
        json.find(R"("calculator.": {"name": "pq", "parameters": {}, )"),
        std::string::npos
    );
    EXPECT_TRUE(json.ends_with(R"(]}}, "info": {"step": 7}})")) << json;

    EXPECT_EQ(
        readArray<std::int64_t>(data, json, "numbers", 3),
        (std::vector<std::int64_t>{8, 1, 0})
    );
    EXPECT_EQ(
        readArray<double>(data, json, "masses", 3),
        (std::vector<double>{15.999, 1.008, 0.0})
    );
    EXPECT_EQ(
        readArray<double>(data, json, "charges", 3),
        (std::vector<double>{-0.8, 0.4, 0.4})
    );

    const auto positions = readArray<double>(data, json, "positions", 9);
    const auto momenta   = readArray<double>(data, json, "momenta", 9);
    const auto forces    = readArray<double>(data, json, "forces", 9);

    for (std::size_t atomIndex = 0; atomIndex < 3; ++atomIndex)
    {
        const auto &atom = ATOMS[atomIndex];
        for (std::size_t dim = 0; dim < 3; ++dim)
        {
            const auto index    = (3 * atomIndex) + dim;
            const auto momentum = atom.mass * atom.velocity[dim] / ASE_FS;
            const auto force    = atom.force[dim] * ASE_KCAL_PER_MOL;

            EXPECT_DOUBLE_EQ(positions[index], atom.position[dim]);
            EXPECT_NEAR(momenta[index], momentum, TOLERANCE);
            EXPECT_NEAR(forces[index], force, TOLERANCE);
        }
    }

    EXPECT_NEAR(jsonValue(json, "energy"), -1.5 * ASE_KCAL_PER_MOL, TOLERANCE);

    const auto second = itemJson(data, 1);
    EXPECT_EQ(second.find("numbers."), std::string::npos);
    EXPECT_EQ(second.find("masses."), std::string::npos);
    EXPECT_TRUE(second.starts_with(R"({"positions.": )")) << second;
    EXPECT_TRUE(second.ends_with(R"("info": {"step": 8}})")) << second;

    std::filesystem::remove(filename);
}

/**
 * @brief the header is written with the first frame
 *
 */
TEST(TestAseTrajOutput, writesNothingBeforeTheFirstFrame)
{
    const std::string filename = "testAseTrajOutputEmpty.traj";

    {
        out::AseTrajOutput output{filename};
        output.setFilename(filename, true);
    }

    EXPECT_TRUE(readFile(filename).empty());

    std::filesystem::remove(filename);
}

/**
 * @brief errors are reported, not silently ignored
 *
 */
TEST(TestAseTrajOutput, throwsOnInvalidUse)
{
    const std::string filename = "testAseTrajOutputErrors.traj";

    {
        out::AseTrajOutput output{filename};
        output.setFilename(filename, true);
        output.write(makeFrame(3), 0);

        EXPECT_THROW_MSG(
            output.write(makeFrame(2), 1),
            exc::InputFileException,
            "ASE trajectory output needs the same number of atoms in every "
            "frame"
        );

        output.close();
        EXPECT_THROW_MSG(
            output.write(makeFrame(3), 2),
            exc::InputFileException,
            "ASE trajectory is closed - filename = " + filename
        );
    }

    out::AseTrajOutput existing{filename};
    EXPECT_THROW_MSG(
        existing.setFilename(filename, false),
        exc::InputFileException,
        "File already exists - filename = " + filename
    );

    std::filesystem::remove(filename);
}

/**
 * @brief a full disk fails the write instead of losing frames
 *
 */
TEST(TestAseTrajOutput, throwsIfWritingFails)
{
    const std::string full = "/dev/full";
    if (!std::filesystem::exists(full))
    {
        GTEST_SKIP() << "needs /dev/full";
    }

    out::AseTrajOutput output{full};
    output.setFilename(full, true);
    EXPECT_THROW_MSG(
        output.write(makeFrame(1), 0),
        exc::InputFileException,
        "Could not write ASE trajectory - filename = " + full
    );
}
