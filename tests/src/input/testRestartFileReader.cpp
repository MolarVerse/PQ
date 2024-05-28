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

#include "testRestartFileReader.hpp"

#include <string>   // for string, allocator, basic_string
#include <vector>   // for vector

#include "fileSettings.hpp"          // for FileSettings
#include "gtest/gtest.h"             // for Message, TestPartResult
#include "moldescriptorReader.hpp"   // for MoldescriptorReader
#include "restartFileReader.hpp"     // for RstFileReader, readRstFile
#include "restartFileSection.hpp"    // for RstFileSection, readInput

using namespace input;

/**
 * @brief tests determineSection base on the first element of the line
 *
 */
TEST_F(TestRstFileReader, determineSection)
{
    std::string                    filename = "examples/setup/h2o_qmcfc.rst";
    restartFile::RestartFileReader rstFileReader(filename, *_engine);

    auto  lineElements = std::vector<std::string>{"sTeP", "1"};
    auto *section      = rstFileReader.determineSection(lineElements);
    EXPECT_EQ(section->keyword(), "step");

    lineElements = std::vector<std::string>{"Box"};
    section      = rstFileReader.determineSection(lineElements);
    EXPECT_EQ(section->keyword(), "box");

    lineElements = std::vector<std::string>{"notAHeaderSection"};
    section      = rstFileReader.determineSection(lineElements);
    EXPECT_EQ(section->keyword(), "");
}

/**
 * @brief test full read restart file function
 *
 */
TEST_F(TestRstFileReader, rstFileReading)
{
    settings::FileSettings::setMolDescriptorFileName(
        "examples/setup/moldescriptor.dat"
    );
    molDescriptor::MoldescriptorReader moldescriptor(*_engine);

    std::string filename = "examples/setup/h2o-qmcf.rst";
    settings::FileSettings::setStartFileName(filename);

    moldescriptor.read();
    ASSERT_NO_THROW(restartFile::readRestartFile(*_engine));
}