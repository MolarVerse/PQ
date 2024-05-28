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

#include "bondSection.hpp"                // for BondSection
#include "bondType.hpp"                   // for BondType
#include "engine.hpp"                     // for Engine
#include "exceptions.hpp"                 // for ParameterFileException
#include "forceFieldClass.hpp"            // for ForceField
#include "parameterFileSection.hpp"       // for parameterFile
#include "testParameterFileSection.hpp"   // for TestParameterFileSection
#include "throwWithMessage.hpp"           // for ASSERT_THROW_MSG

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <gtest/gtest.h>   // for EXPECT_EQ, TestInfo (ptr only)
#include <string>          // for string, allocator, basic_string
#include <vector>          // for vector

using namespace input::parameterFile;

/**
 * @brief test bonds section processing one line
 *
 */
TEST_F(TestParameterFileSection, processSectionBonds)
{
    std::vector<std::string>              lineElements = {"0", "1.22", "234.3"};
    input::parameterFile::BondSection bondSection;
    bondSection.processSection(lineElements, *_engine);
    EXPECT_EQ(_engine->getForceField().getBondTypes().size(), 1);
    EXPECT_EQ(_engine->getForceField().getBondTypes()[0].getId(), 0);
    EXPECT_EQ(_engine->getForceField().getBondTypes()[0].getEquilibriumBondLength(), 1.22);
    EXPECT_EQ(_engine->getForceField().getBondTypes()[0].getForceConstant(), 234.3);

    lineElements = {"1", "2", "1.0", "0"};
    EXPECT_THROW(bondSection.processSection(lineElements, *_engine), customException::ParameterFileException);

    lineElements = {"1", "-2", "1.0"};
    EXPECT_THROW(bondSection.processSection(lineElements, *_engine), customException::ParameterFileException);
}

TEST_F(TestParameterFileSection, endedNormallyBonds)
{
    auto bondSection = BondSection();
    ASSERT_NO_THROW(bondSection.endedNormally(true));

    ASSERT_THROW_MSG(bondSection.endedNormally(false),
                     customException::ParameterFileException,
                     "Parameter file bonds section ended abnormally!");
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}