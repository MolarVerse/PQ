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

#include "angleSection.hpp"
#include "bondSection.hpp"
#include "dihedralSection.hpp"
#include "exceptions.hpp"
#include "fileSettings.hpp"
#include "forceFieldNonCoulomb.hpp"
#include "forceFieldSettings.hpp"
#include "improperDihedralSection.hpp"
#include "jCouplingSection.hpp"
#include "nonCoulombicsSection.hpp"
#include "parameterFileReader.hpp"
#include "throwWithMessage.hpp"
#include "typesSection.hpp"

/**
 * @brief tests determine section function
 *
 */
TEST_F(TestParameterFileReader, determineSection)
{
    EXPECT_EQ(_parameterFileReader->getParameterFileSections().size(), 7);

    const auto *section = _parameterFileReader->determineSection({"types"});
    EXPECT_EQ(typeid(*section), typeid(input::parameterFile::TypesSection));

    const auto *section1 = _parameterFileReader->determineSection({"bonds"});
    EXPECT_EQ(typeid(*section1), typeid(input::parameterFile::BondSection));

    const auto *section2 = _parameterFileReader->determineSection({"angles"});
    EXPECT_EQ(typeid(*section2), typeid(input::parameterFile::AngleSection));

    const auto *section3 =
        _parameterFileReader->determineSection({"dihedrals"});
    EXPECT_EQ(typeid(*section3), typeid(input::parameterFile::DihedralSection));

    const auto *section4 =
        _parameterFileReader->determineSection({"impropers"});
    EXPECT_EQ(
        typeid(*section4),
        typeid(input::parameterFile::ImproperDihedralSection)
    );

    const auto *section5 =
        _parameterFileReader->determineSection({"nonCoulombics"});
    EXPECT_EQ(
        typeid(*section5),
        typeid(input::parameterFile::NonCoulombicsSection)
    );

    const auto *section6 =
        _parameterFileReader->determineSection({"j_couplings"});
    EXPECT_EQ(
        typeid(*section6),
        typeid(input::parameterFile::JCouplingSection)
    );

    EXPECT_THROW_MSG(
        [[maybe_unused]] const auto dummy =
            _parameterFileReader->determineSection({"N.A."}),
        exc::ParameterFileException,
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
    EXPECT_EQ(_parameterFileReader->getParameterFileSections().size(), 7);

    const auto *section = _parameterFileReader->determineSection({"types"});
    _parameterFileReader->deleteSection(section);
    EXPECT_EQ(_parameterFileReader->getParameterFileSections().size(), 6);

    const auto *section1 = _parameterFileReader->determineSection({"bonds"});
    _parameterFileReader->deleteSection(section1);
    EXPECT_EQ(_parameterFileReader->getParameterFileSections().size(), 5);

    const auto *section2 = _parameterFileReader->determineSection({"angles"});
    _parameterFileReader->deleteSection(section2);
    EXPECT_EQ(_parameterFileReader->getParameterFileSections().size(), 4);

    const auto *section3 =
        _parameterFileReader->determineSection({"dihedrals"});
    _parameterFileReader->deleteSection(section3);
    EXPECT_EQ(_parameterFileReader->getParameterFileSections().size(), 3);

    const auto *section4 =
        _parameterFileReader->determineSection({"impropers"});
    _parameterFileReader->deleteSection(section4);
    EXPECT_EQ(_parameterFileReader->getParameterFileSections().size(), 2);

    const auto *section5 =
        _parameterFileReader->determineSection({"nonCoulombics"});
    _parameterFileReader->deleteSection(section5);
    EXPECT_EQ(_parameterFileReader->getParameterFileSections().size(), 1);

    const auto *section6 =
        _parameterFileReader->determineSection({"j_couplings"});
    _parameterFileReader->deleteSection(section6);
    EXPECT_EQ(_parameterFileReader->getParameterFileSections().size(), 0);
}

/**
 * @brief tests read function
 *
 * @details if filename is empty, exception should be thrown
 *
 */
TEST_F(TestParameterFileReader, readFileNameEmpty)
{
    _settings.forceFieldSettings.setType(ForceFieldType::ON);
    settings::FileSettings::unsetIsParameterFileNameSet();
    EXPECT_THROW_MSG(
        _parameterFileReader->read(),
        exc::InputFileException,
        "Parameter file needed for requested simulation setup"
    );
}

/**
 * @brief tests that reading a non-existent file does not throw and simply
 * leaves all sections unconsumed, since the underlying std::ifstream never
 * reports failure to the caller
 */
TEST_F(TestParameterFileReader, readFileDoesNotExist)
{
    settings::FileSettings::setIsParameterFileNameSet();
    auto reader = input::parameterFile::ParameterFileReader(
        "data/parameterFileReader/doesNotExist.param",
        *_engine
    );
    EXPECT_NO_THROW(reader.read());
    EXPECT_EQ(reader.getParameterFileSections().size(), 7);
}

/**
 * @brief tests read function successfully parses all sections and that
 * every section is consumed exactly once (no section left over)
 */
TEST_F(TestParameterFileReader, readParameterFile)
{
    settings::FileSettings::setIsParameterFileNameSet();
    _engine->getPotential()->makeNonCoulombPotential(
        pot::ForceFieldNonCoulomb()
    );
    EXPECT_NO_THROW(_parameterFileReader->read());

    EXPECT_EQ(_parameterFileReader->getParameterFileSections().size(), 2);

    const auto &remaining = _parameterFileReader->getParameterFileSections();
    for (const auto &section : remaining)
        EXPECT_NE(section->keyword(), "bonds");
}

/**
 * @brief tests that reading a parameter file with a duplicated section
 * keyword throws, since the section is deleted after being processed once
 */
TEST_F(TestParameterFileReader, readParameterFileDuplicateSection)
{
    settings::FileSettings::setIsParameterFileNameSet();
    auto reader = input::parameterFile::ParameterFileReader(
        "data/parameterFileReader/paramDuplicateSection.param",
        *_engine
    );
    EXPECT_THROW_MSG(
        reader.read(),
        exc::ParameterFileException,
        "Unknown or already parsed keyword \"BONDS\" in parameter file"
    );
}

/**
 * @brief tests that reading a parameter file with an unknown section
 * keyword throws
 */
TEST_F(TestParameterFileReader, readParameterFileUnknownSection)
{
    settings::FileSettings::setIsParameterFileNameSet();
    auto reader = input::parameterFile::ParameterFileReader(
        "data/parameterFileReader/paramUnknownSection.param",
        *_engine
    );
    EXPECT_THROW_MSG(
        reader.read(),
        exc::ParameterFileException,
        "Unknown or already parsed keyword \"N.A.\" in parameter file"
    );
}

/**
 * @brief tests that lines starting with a comment character are ignored
 * and the rest of the line is still parsed correctly
 */
TEST_F(TestParameterFileReader, readParameterFileWithComments)
{
    settings::FileSettings::setIsParameterFileNameSet();
    _engine->getPotential()->makeNonCoulombPotential(
        pot::ForceFieldNonCoulomb()
    );
    auto reader = input::parameterFile::ParameterFileReader(
        "data/parameterFileReader/paramWithComments.param",
        *_engine
    );
    EXPECT_NO_THROW(reader.read());
    EXPECT_EQ(reader.getParameterFileSections().size(), 2);
}

/**
 * @brief tests the free readParameterFile function end to end, including
 * that it is a no-op when the force field is not active
 */
TEST_F(TestParameterFileReader, readParameterFileFreeFunction)
{
    _settings.forceFieldSettings.setType(ForceFieldType::ON);
    _engine->getPotential()->makeNonCoulombPotential(
        pot::ForceFieldNonCoulomb()
    );
    settings::FileSettings::setParameterFileName(
        "data/parameterFileReader/param.param"
    );
    EXPECT_NO_THROW(input::parameterFile::readParameterFile(*_engine));
}

TEST_F(TestParameterFileReader, nameNotSetButNotNeeded)
{
    _settings.forceFieldSettings.setType(ForceFieldType::OFF);
    settings::FileSettings::unsetIsParameterFileNameSet();
    EXPECT_NO_THROW(input::parameterFile::readParameterFile(*_engine));
}

/**
 * @brief tests setFilename and getFilename getter/setter
 */
TEST_F(TestParameterFileReader, setAndGetFilename)
{
    EXPECT_EQ(
        _parameterFileReader->getFilename(),
        "data/parameterFileReader/param.param"
    );

    _parameterFileReader->setFilename("newFileName.param");
    EXPECT_EQ(_parameterFileReader->getFilename(), "newFileName.param");
}
