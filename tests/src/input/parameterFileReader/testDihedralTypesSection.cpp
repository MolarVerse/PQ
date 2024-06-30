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

#include "constants/conversionFactors.hpp"   // for _DEG_TO_RAD_
#include "dihedralSection.hpp"               // for DihedralSection
#include "dihedralType.hpp"                  // for DihedralType
#include "engine.hpp"                        // for Engine
#include "exceptions.hpp"                    // for ParameterFileException
#include "forceFieldClass.hpp"               // for ForceField
#include "parameterFileSection.hpp"          // for parameterFile
#include "testParameterFileSection.hpp"      // for TestParameterFileSection
#include "throwWithMessage.hpp"              // for ASSERT_THROW_MSG

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <gtest/gtest.h>   // for EXPECT_EQ, TestInfo (ptr only)
#include <string>          // for string, allocator, basic_string
#include <vector>          // for vector

using namespace input::parameterFile;

/**
 * @brief test bonds section processing one line
 *
 */
TEST_F(TestParameterFileSection, processSectionDihedral)
{
    std::vector<std::string> lineElements = {"0", "1.22", "234.3", "324.3"};
    DihedralSection          dihedralSection;
    dihedralSection.processSection(lineElements, *_engine);
    EXPECT_EQ(_engine->getForceField().getDihedralTypes().size(), 1);
    EXPECT_EQ(_engine->getForceField().getDihedralTypes()[0].getId(), 0);
    EXPECT_EQ(_engine->getForceField().getDihedralTypes()[0].getForceConstant(), 1.22);
    EXPECT_EQ(_engine->getForceField().getDihedralTypes()[0].getPeriodicity(), 234.3);
    EXPECT_EQ(_engine->getForceField().getDihedralTypes()[0].getPhaseShift(), 324.3 * constants::_DEG_TO_RAD_);

    lineElements = {"1", "2", "1.0", "0", "2"};
    EXPECT_THROW(dihedralSection.processSection(lineElements, *_engine), customException::ParameterFileException);

    lineElements = {"1", "2", "-1.0", "3"};
    EXPECT_THROW(dihedralSection.processSection(lineElements, *_engine), customException::ParameterFileException);
}

TEST_F(TestParameterFileSection, endedNormallyDihedral)
{
    auto dihedralSection = DihedralSection();
    ASSERT_NO_THROW(dihedralSection.endedNormally(true));

    ASSERT_THROW_MSG(dihedralSection.endedNormally(false),
                     customException::ParameterFileException,
                     "Parameter file dihedrals section ended abnormally!");
}