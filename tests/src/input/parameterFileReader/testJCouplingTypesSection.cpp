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

#include <string>
#include <vector>

#include "engine.hpp"
#include "exceptions.hpp"
#include "jCouplingSection.hpp"
#include "testParameterFileSection.hpp"

using namespace input::parameterFile;
using namespace customException;

TEST_F(TestParameterFileSection, jCouplingSectionKeyword)
{
    JCouplingSection section;
    EXPECT_EQ(section.keyword(), "j_couplings");
}

TEST_F(TestParameterFileSection, jCouplingSectionProcessSevenElements)
{
    JCouplingSection section;
    // id, J0, fc, a, b, c, phase
    std::vector<std::string> lineElements =
        {"7", "1.0", "2.0", "3.0", "4.0", "5.0", "30.0"};
    section.processSection(lineElements, *_engine);

    const auto &types = _engine->getForceField()->getJCouplTypes();
    ASSERT_EQ(types.size(), 1u);
    EXPECT_EQ(types[0].getId(), 7u);
    EXPECT_DOUBLE_EQ(types[0].getJ0(), 1.0);
    EXPECT_DOUBLE_EQ(types[0].getForceConstant(), 2.0);
    EXPECT_DOUBLE_EQ(types[0].getA(), 3.0);
    EXPECT_DOUBLE_EQ(types[0].getB(), 4.0);
    EXPECT_DOUBLE_EQ(types[0].getC(), 5.0);
}

TEST_F(TestParameterFileSection, jCouplingSectionAcceptsZeroSymmetry)
{
    JCouplingSection         section;
    std::vector<std::string> lineElements =
        {"1", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0"};
    EXPECT_NO_THROW(section.processSection(lineElements, *_engine));
    EXPECT_EQ(_engine->getForceField()->getJCouplTypes().size(), 1u);
}

TEST_F(TestParameterFileSection, jCouplingSectionAcceptsPlusSymmetry)
{
    JCouplingSection         section;
    std::vector<std::string> lineElements =
        {"2", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "+"};
    EXPECT_NO_THROW(section.processSection(lineElements, *_engine));
}

TEST_F(TestParameterFileSection, jCouplingSectionAcceptsMinusSymmetry)
{
    JCouplingSection         section;
    std::vector<std::string> lineElements =
        {"3", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "-"};
    EXPECT_NO_THROW(section.processSection(lineElements, *_engine));
}

TEST_F(TestParameterFileSection, jCouplingSectionThrowsOnTooFewElements)
{
    JCouplingSection         section;
    std::vector<std::string> lineElements = {"1", "2", "3"};
    EXPECT_THROW(
        section.processSection(lineElements, *_engine),
        ParameterFileException
    );
}

TEST_F(TestParameterFileSection, jCouplingSectionThrowsOnTooManyElements)
{
    JCouplingSection         section;
    std::vector<std::string> lineElements =
        {"1", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "+", "extra"};
    EXPECT_THROW(
        section.processSection(lineElements, *_engine),
        ParameterFileException
    );
}
