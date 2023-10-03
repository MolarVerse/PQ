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

#include "testParameterFileSection.hpp"

#include "bondSection.hpp"   // for BondSection

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <ostream>         // for operator<<, ofstream, basic_ostream, endl
#include <vector>          // for vector

/**
 * @brief tests full process function TODO: think of a clever way to test this
 *
 */
TEST_F(TestParameterFileSection, processParameterSection)
{
    input::parameterFile::BondSection section;

    std::ofstream outputStream(_parameterFileName.c_str());

    outputStream << "bonds\n";
    outputStream << "1 2 1.0\n";
    outputStream << "         \n";
    outputStream << "2 3 1.2\n";
    outputStream << "end" << '\n' << std::flush;

    outputStream.close();

    auto          lineElements = std::vector{std::string("")};
    std::ifstream fp(_parameterFileName.c_str());
    getline(fp, lineElements[0]);

    section.setFp(&fp);
    section.setLineNumber(1);

    EXPECT_NO_THROW(section.process(lineElements, *_engine));

    EXPECT_EQ(section.getLineNumber(), 5);
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}