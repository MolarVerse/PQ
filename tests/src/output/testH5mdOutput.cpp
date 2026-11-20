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

#include "atom.hpp"
#include "exceptions.hpp"
#include "frame.hpp"
#include "h5mdOutput.hpp"
#include "molecule.hpp"
#include "orthorhombicBox.hpp"
#include "simulationBox.hpp"
#include "throwWithMessage.hpp"
#include "units.hpp"

namespace
{
    out::Frame makeFrame()
    {
        molsys::SimulationBox simulationBox;
        simulationBox.setBox(
            molsys::OrthorhombicBox{
                {10.0 * angstrom, 11.0 * angstrom, 12.0 * angstrom}
            },
            false
        );

        molsys::Molecule molecule{MolType{1}};
        for (const auto *name : {"O", "D", "dum"})
        {
            auto atom = std::make_shared<molsys::Atom>();
            atom->setName(name);
            atom->setPosition({1.0 * angstrom, 2.0 * angstrom, 3.0 * angstrom});
            molecule.addAtom(atom);
            simulationBox.addAtom(atom);
        }
        simulationBox.addMolecule(molecule);

        return out::Frame{simulationBox};
    }
}   // namespace

/**
 * @brief without HDF5 the H5MD output writes nothing; with it, a file is
 * written (its content is checked in testH5mdOutputContent)
 *
 */
TEST(TestH5mdOutput, writesOnlyWithHdf5)
{
    const std::string filename = "testH5mdOutput.h5md";
    std::filesystem::remove(filename);

    if (!out::H5mdOutput::available())
    {
        {
            out::H5mdOutput output{filename, true, 0.5};
            output.write(makeFrame(), 10);
            output.close();
        }
        EXPECT_FALSE(std::filesystem::exists(filename));
        return;
    }

    {
        out::H5mdOutput output{filename, false, 0.5};
        output.write(makeFrame(), 10);
        output.close();
    }

    EXPECT_TRUE(std::filesystem::exists(filename));
    EXPECT_THROW_MSG(
        out::H5mdOutput(filename, false, 0.5),
        exc::InputFileException,
        "File already exists - filename = " + filename
    );

    std::filesystem::remove(filename);
}

/**
 * @brief species is the atomic number, 0 for atoms without an element
 *
 */
TEST(TestH5mdOutput, species)
{
    EXPECT_EQ(out::H5mdOutput::species("O"), 8);
    EXPECT_EQ(out::H5mdOutput::species("h"), 1);
    EXPECT_EQ(out::H5mdOutput::species("dum"), 0);
    EXPECT_EQ(out::H5mdOutput::species("cav"), 0);
    EXPECT_EQ(out::H5mdOutput::species("notAnElement"), 0);
}
