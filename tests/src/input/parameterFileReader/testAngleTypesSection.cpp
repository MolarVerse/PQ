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

#include <gtest/gtest.h>   // for EXPECT_EQ, TestInfo (ptr only)

#include <string>   // for string, allocator, basic_string
#include <vector>   // for vector

#include "angleSection.hpp"                  // for AngleSection
#include "constants/conversionFactors.hpp"   // for _DEG_TO_RAD_
#include "engine.hpp"                        // for Engine
#include "exceptions.hpp"                    // for ParameterFileException
#include "gtest/gtest.h"                     // for Message, TestPartResult
#include "testParameterFileSection.hpp"      // for TestParameterFileSection
#include "throwWithMessage.hpp"              // for ASSERT_THROW_MSG

using namespace input::parameterFile;

/**
 * @brief test bonds section processing one line
 *
 */
TEST_F(TestParameterFileSection, processSectionAngle)
{
    std::vector<std::string>           lineElements = {"0", "1.22", "234.3"};
    input::parameterFile::AngleSection angleSection;
    angleSection.processSection(lineElements, *_engine);

    const auto &angleTypes = _engine->getForceField()->getAngleTypes();

    EXPECT_EQ(angleTypes.size(), 1);
    EXPECT_EQ(angleTypes[0].getId(), AngleId{0});
    EXPECT_EQ(
        angleTypes[0].getEquilibriumAngle(),
        1.22 * constants::DEG_TO_RAD
    );
    EXPECT_EQ(angleTypes[0].getForceConstant(), 234.3);

    lineElements = {"1", "2", "1.0", "0"};
    EXPECT_THROW(
        angleSection.processSection(lineElements, *_engine),
        exc::ParameterFileException
    );
}

TEST_F(TestParameterFileSection, endedNormallyAngle)
{
    auto angleSection = AngleSection();
    ASSERT_NO_THROW(angleSection.endedNormally(true));

    ASSERT_THROW_MSG(
        angleSection.endedNormally(false),
        exc::ParameterFileException,
        "Parameter file angles section ended abnormally!"
    );
}
