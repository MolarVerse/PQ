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

#include <gtest/gtest.h>   // for EXPECT_THROW, TestInfo (ptr ...

#include <string>   // for string, allocator, basic_string
#include <vector>   // for vector

#include "exceptions.hpp"                 // for ParameterFileException
#include "gtest/gtest.h"                  // for Message, TestPartResult, tes...
#include "parameterFileSection.hpp"       // for parameterFile
#include "potentialSettings.hpp"          // for PotentialSettings
#include "testParameterFileSection.hpp"   // for TestParameterFileSection
#include "throwWithMessage.hpp"           // for ASSERT_THROW_MSG
#include "typesSection.hpp"               // for TypesSection

using namespace input::parameterFile;

/**
 * @brief test types section processing one line
 *
 */
TEST_F(TestParameterFileSection, processSectionTypes)
{
    std::vector<std::string> lineElements =
        {"1", "2", "1.0", "0", "s", "f", "0.23", "0.99"};
    input::parameterFile::TypesSection typesSection;
    typesSection.process(lineElements, *_engine);
    EXPECT_EQ(settings::PotentialSettings::getScale14Coulomb(), 0.23);
    EXPECT_EQ(settings::PotentialSettings::getScale14VDW(), 0.99);

    lineElements = {"1", "2", "1.0", "0", "s", "f", "0.23"};
    EXPECT_THROW(
        typesSection.process(lineElements, *_engine),
        customException::ParameterFileException
    );

    lineElements = {"1", "2", "1.0", "0", "s", "f", "0.23", "1.01"};
    EXPECT_THROW(
        typesSection.process(lineElements, *_engine),
        customException::ParameterFileException
    );

    lineElements = {"1", "2", "1.0", "0", "s", "f", "1.23", "0.01"};
    EXPECT_THROW(
        typesSection.process(lineElements, *_engine),
        customException::ParameterFileException
    );

    lineElements = {"1", "2", "1.0", "0", "s", "f", "-0.23", "0.01"};
    EXPECT_THROW(
        typesSection.process(lineElements, *_engine),
        customException::ParameterFileException
    );

    lineElements = {"1", "2", "1.0", "0", "s", "f", "0.23", "-0.01"};
    EXPECT_THROW(
        typesSection.process(lineElements, *_engine),
        customException::ParameterFileException
    );
}

TEST_F(TestParameterFileSection, endedNormallyTypes)
{
    auto typesSection = TypesSection();
    ASSERT_NO_THROW(typesSection.endedNormally(true));

    ASSERT_THROW_MSG(
        typesSection.endedNormally(false),
        customException::ParameterFileException,
        "Parameter file types section ended abnormally!"
    );
}

/**
 * @brief just a dummy test for processHeader of types section as it is not used
 *
 */
TEST_F(TestParameterFileSection, dummyHeaderTest)
{
    auto typesSection = TypesSection();
    auto lineElements = std::vector<std::string>({"dummy"});
    EXPECT_NO_THROW(typesSection.processHeader(lineElements, *_engine));
}