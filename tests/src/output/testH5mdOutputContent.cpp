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

// only built with BUILD_WITH_HDF5; empty for tools without the HDF5 headers
#include "units.hpp"
#if __has_include(<hdf5.h>)

#include <gtest/gtest.h>
#include <hdf5.h>

#include <array>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "atom.hpp"
#include "frame.hpp"
#include "h5mdOutput.hpp"
#include "molecule.hpp"
#include "orthorhombicBox.hpp"
#include "simulationBox.hpp"

namespace
{
    out::Frame makeFrame(double potentialEnergy)
    {
        molsys::SimulationBox simulationBox;
        simulationBox.setBox(
            molsys::OrthorhombicBox{
                {10.0 * angstrom, 11.0 * angstrom, 12.0 * angstrom}
            },
            false
        );

        molsys::Molecule                  molecule{MolType{1}};
        const std::array<const char *, 3> names{"O", "D", "dum"};
        for (std::size_t i = 0; i < names.size(); ++i)
        {
            const auto x    = static_cast<double>(i);
            auto       atom = std::make_shared<molsys::Atom>();
            atom->setName(names.at(i));
            atom->setPosition(
                {(x + 1.0) * angstrom, 2.0 * angstrom, 3.0 * angstrom}
            );
            atom->setVelocity(
                {0.0 * angstrom / s,
                 x * 1.0e12 * angstrom / s,
                 0.0 * angstrom / s}
            );
            atom->setForce(
                {0.0 * kcal_per_mol / angstrom,
                 0.0 * kcal_per_mol / angstrom,
                 -x * kcal_per_mol / angstrom}
            );
            atom->setPartialCharge(0.5 * x * elementary_charge);
            molecule.addAtom(atom);
            simulationBox.addAtom(atom);
        }
        simulationBox.addMolecule(molecule);

        auto frame            = out::Frame{simulationBox};
        frame.potentialEnergy = potentialEnergy * kcal_per_mol;
        return frame;
    }

    // reads a whole dataset as double; returns its dimensions in dims
    std::vector<double> readDoubles(
        hid_t                 file,
        const char           *path,
        std::vector<hsize_t> &dims
    )
    {
        const hid_t set   = H5Dopen2(file, path, H5P_DEFAULT);
        const hid_t space = H5Dget_space(set);
        dims.resize(
            static_cast<std::size_t>(H5Sget_simple_extent_ndims(space))
        );
        H5Sget_simple_extent_dims(space, dims.data(), nullptr);

        std::size_t size = 1;
        for (const auto extent : dims) size *= extent;

        std::vector<double> values(size);
        H5Dread(
            set,
            H5T_NATIVE_DOUBLE,
            H5S_ALL,
            H5S_ALL,
            H5P_DEFAULT,
            values.data()
        );
        H5Sclose(space);
        H5Dclose(set);
        return values;
    }

    std::string readUnit(hid_t file, const char *path)
    {
        const hid_t set  = H5Dopen2(file, path, H5P_DEFAULT);
        const hid_t attr = H5Aopen(set, "unit", H5P_DEFAULT);
        const hid_t type = H5Aget_type(attr);

        char *data = nullptr;
        H5Aread(attr, type, static_cast<void *>(&data));
        std::string unit{data};

        const hid_t space = H5Aget_space(attr);
        H5Dvlen_reclaim(type, space, H5P_DEFAULT, static_cast<void *>(&data));
        H5Sclose(space);
        H5Tclose(type);
        H5Aclose(attr);
        H5Dclose(set);
        return unit;
    }
}   // namespace

/**
 * @brief two frames in H5MD layout, read back with the HDF5 library
 *
 */
TEST(TestH5mdOutputContent, writesH5mdLayout)
{
    const std::string filename = "testH5mdOutputContent.h5md";
    std::filesystem::remove(filename);

    {
        out::H5mdOutput output{filename, false, 0.5};
        output.write(makeFrame(-1.5), 10);
        output.write(makeFrame(-2.5), 20);
    }

    const hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    std::vector<hsize_t> dims;

    const auto species = readDoubles(file, "/particles/all/species", dims);
    EXPECT_EQ(species, (std::vector<double>{8.0, 1.0, 0.0}));

    const auto position =
        readDoubles(file, "/particles/all/position/value", dims);
    EXPECT_EQ(dims, (std::vector<hsize_t>{2, 3, 3}));
    EXPECT_EQ(position[3], 2.0);
    EXPECT_EQ(position[17], 3.0);

    const auto velocity =
        readDoubles(file, "/particles/all/velocity/value", dims);
    EXPECT_FLOAT_EQ(static_cast<float>(velocity[7]), 2.0e-3F);   // A/fs

    const auto force = readDoubles(file, "/particles/all/force/value", dims);
    EXPECT_EQ(force[8], -2.0);

    const auto charge = readDoubles(file, "/particles/all/charge/value", dims);
    EXPECT_EQ(dims, (std::vector<hsize_t>{2, 3}));
    EXPECT_EQ(charge[2], 1.0);

    const auto steps = readDoubles(file, "/particles/all/position/step", dims);
    EXPECT_EQ(steps, (std::vector<double>{10.0, 20.0}));

    const auto times = readDoubles(file, "/particles/all/position/time", dims);
    EXPECT_EQ(times, (std::vector<double>{5.0, 10.0}));

    const auto edges =
        readDoubles(file, "/particles/all/box/edges/value", dims);
    EXPECT_EQ(dims, (std::vector<hsize_t>{2, 3, 3}));
    EXPECT_EQ(edges[0], 10.0);
    EXPECT_EQ(edges[4], 11.0);
    EXPECT_EQ(edges[8], 12.0);

    const auto energy =
        readDoubles(file, "/observables/potential_energy/value", dims);
    EXPECT_EQ(energy, (std::vector<double>{-1.5, -2.5}));

    EXPECT_EQ(readUnit(file, "/particles/all/position/value"), "Angstrom");
    EXPECT_EQ(readUnit(file, "/particles/all/velocity/value"), "Angstrom fs-1");
    EXPECT_EQ(
        readUnit(file, "/observables/potential_energy/value"),
        "kcal mol-1"
    );

    H5Fclose(file);
    std::filesystem::remove(filename);
}

#endif   // __has_include(<hdf5.h>)
