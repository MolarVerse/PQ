/*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
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

#include "engine.hpp"                   // for Engine
#include "exceptions.hpp"               // for RstFileException, customException
#include "restartFileSection.hpp"       // for RstFileSection, readInput
#include "settings.hpp"                 // for Settings
#include "simulationBox.hpp"            // for SimulationBox
#include "simulationBoxSettings.hpp"    // for SimulationBoxSettings
#include "testRestartFileSection.hpp"   // for TestBoxSection

#include "gmock/gmock.h"   // for ElementsAre, MakePredicateForma...
#include "gtest/gtest.h"   // for Message, TestPartResult, Assert...
#include <cstddef>         // for size_t, std
#include <gtest/gtest.h>   // for TestInfo (ptr only), ASSERT_THROW
#include <string>          // for string, allocator, basic_string
#include <vector>          // for vector

using namespace input;

TEST_F(TestBoxSection, testKeyword) { EXPECT_EQ(_section->keyword(), "box"); }

TEST_F(TestBoxSection, testIsHeader) { EXPECT_TRUE(_section->isHeader()); }

TEST_F(TestBoxSection, testNumberOfArguments)
{
    for (size_t i = 0; i < 10; ++i)
        if (i != 4 && i != 7)
        {
            auto line = std::vector<std::string>(i);
            ASSERT_THROW(_section->process(line, *_engine), customException::RstFileException);
        }
}

TEST_F(TestBoxSection, testProcess)
{
    settings::Settings::setJobtype(settings::JobType::QM_MD);

    EXPECT_EQ(settings::SimulationBoxSettings::getBoxSet(), false);

    std::vector<std::string> line = {"box", "1.0", "2.0", "3.0"};
    _section->process(line, *_engine);
    ASSERT_THAT(_engine->getSimulationBox().getBoxDimensions(), testing::ElementsAre(1.0, 2.0, 3.0));
    ASSERT_THAT(_engine->getSimulationBox().getBoxAngles(), testing::ElementsAre(90.0, 90.0, 90.0));

    line = {"box", "1.0", "2.0", "3.0", "90.0", "90.0", "70.0"};
    _section->process(line, *_engine);
    ASSERT_THAT(_engine->getSimulationBox().getBoxDimensions(), testing::ElementsAre(1.0, 2.0, 3.0));
    ASSERT_THAT(_engine->getSimulationBox().getBoxAngles(), testing::ElementsAre(90.0, 90.0, 70.0));

    line = {"box", "1.0", "2.0", "-3.0", "90.0", "90.0", "90.0"};
    ASSERT_THROW(_section->process(line, *_engine), customException::RstFileException);

    line = {"box", "1.0", "2.0", "3.0", "90.0", "90.0", "190.0"};
    ASSERT_THROW(_section->process(line, *_engine), customException::RstFileException);

    line = {"box", "1.0", "2.0", "3.0", "90.0", "90.0", "-90.0"};
    ASSERT_THROW(_section->process(line, *_engine), customException::RstFileException);

    EXPECT_EQ(settings::SimulationBoxSettings::getBoxSet(), true);
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}