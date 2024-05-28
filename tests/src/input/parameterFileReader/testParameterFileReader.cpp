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

#include "testParameterFileReader.hpp"

#include <vector>   // for vector

#include "angleSection.hpp"      // for AngleSection
#include "bondSection.hpp"       // for BondSection
#include "dihedralSection.hpp"   // for DihedralSection
#include "exceptions.hpp"     // for InputFileException, ParameterFileException
#include "fileSettings.hpp"   // for FileSettings
#include "forceFieldNonCoulomb.hpp"   // for ForceFieldNonCoulomb
#include "forceFieldSettings.hpp"     // for ForceFieldSettings
#include "gtest/gtest.h"   // for Message, TestPartResult, AssertHelper, Test
#include "improperDihedralSection.hpp"   // for ImproperDihedralSection
#include "jCouplingSection.hpp"          // for JCouplingSection
#include "nonCoulombicsSection.hpp"      // for NonCoulombicsSection
#include "parameterFileReader.hpp"       // for ParameterFileReader
#include "potential.hpp"                 // for Potential
#include "throwWithMessage.hpp"          // for EXPECT_THROW_MSG
#include "typesSection.hpp"              // for TypesSection

using namespace input::parameterFile;

/**
 * @brief tests isNeeded function
 *
 * @return true if forceField is enabled
 * @return false
 */
TEST_F(TestParameterFileReader, isNeeded)
{
    EXPECT_FALSE(isNeeded());

    settings::ForceFieldSettings::activate();
    EXPECT_TRUE(isNeeded());
}

/**
 * @brief tests determine section function
 *
 */
TEST_F(TestParameterFileReader, determineSection)
{
    auto *reader = _parameterFileReader;
    EXPECT_EQ(reader->getParameterFileSections().size(), 7);

    const auto *section = reader->determineSection({"types"});
    EXPECT_EQ(typeid(*section), typeid(input::parameterFile::TypesSection));

    const auto *section1 = reader->determineSection({"bonds"});
    EXPECT_EQ(typeid(*section1), typeid(input::parameterFile::BondSection));

    const auto *section2 = reader->determineSection({"angles"});
    EXPECT_EQ(typeid(*section2), typeid(input::parameterFile::AngleSection));

    const auto *section3 = reader->determineSection({"dihedrals"});
    EXPECT_EQ(typeid(*section3), typeid(input::parameterFile::DihedralSection));

    const auto *section4 = reader->determineSection({"impropers"});
    EXPECT_EQ(
        typeid(*section4),
        typeid(input::parameterFile::ImproperDihedralSection)
    );

    const auto *section5 = reader->determineSection({"nonCoulombics"});
    EXPECT_EQ(
        typeid(*section5),
        typeid(input::parameterFile::NonCoulombicsSection)
    );

    const auto *section6 = reader->determineSection({"j-couplings"});
    EXPECT_EQ(
        typeid(*section6),
        typeid(input::parameterFile::JCouplingSection)
    );

    EXPECT_THROW_MSG(
        [[maybe_unused]] const auto dummy = reader->determineSection({"N.A."}),
        customException::ParameterFileException,
        "Unknown or already parsed keyword \"N.A.\" in parameter file"
    );
}

/**
 * @brief tests delete section function
 *
 * @details if section is found, section should be deleted
 *
 */
TEST_F(TestParameterFileReader, deleteSection)
{
    auto *reader = _parameterFileReader;
    EXPECT_EQ(reader->getParameterFileSections().size(), 7);

    const auto *section = reader->determineSection({"types"});
    reader->deleteSection(section);
    EXPECT_EQ(reader->getParameterFileSections().size(), 6);

    const auto *section1 = reader->determineSection({"bonds"});
    reader->deleteSection(section1);
    EXPECT_EQ(reader->getParameterFileSections().size(), 5);

    const auto *section2 = reader->determineSection({"angles"});
    reader->deleteSection(section2);
    EXPECT_EQ(reader->getParameterFileSections().size(), 4);

    const auto *section3 = reader->determineSection({"dihedrals"});
    reader->deleteSection(section3);
    EXPECT_EQ(reader->getParameterFileSections().size(), 3);

    const auto *section4 = reader->determineSection({"impropers"});
    reader->deleteSection(section4);
    EXPECT_EQ(reader->getParameterFileSections().size(), 2);

    const auto *section5 = reader->determineSection({"nonCoulombics"});
    reader->deleteSection(section5);
    EXPECT_EQ(reader->getParameterFileSections().size(), 1);

    const auto *section6 = reader->determineSection({"j-couplings"});
    reader->deleteSection(section6);
    EXPECT_EQ(reader->getParameterFileSections().size(), 0);
}

/**
 * @brief tests read function
 *
 * @details if filename is empty, exception should be thrown
 *
 */
TEST_F(TestParameterFileReader, read_fileNameEmpty)
{
    settings::ForceFieldSettings::activate();
    settings::FileSettings::unsetIsParameterFileNameSet();
    EXPECT_THROW_MSG(
        _parameterFileReader->read(),
        customException::InputFileException,
        "Parameter file needed for requested simulation setup"
    );
}

TEST_F(TestParameterFileReader, readParameterFile)
{
    settings::ForceFieldSettings::activate();
    _engine->getPotential().makeNonCoulombPotential(
        potential::ForceFieldNonCoulomb()
    );
    settings::FileSettings::setParameterFileName(
        "data/parameterFileReader/param.param"
    );
    EXPECT_NO_THROW(input::parameterFile::readParameterFile(*_engine));
}

TEST_F(TestParameterFileReader, nameNotSetButNotNeeded)
{
    settings::ForceFieldSettings::deactivate();
    settings::FileSettings::unsetIsParameterFileNameSet();
    EXPECT_NO_THROW(input::parameterFile::readParameterFile(*_engine));
}
