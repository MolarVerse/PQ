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

#include <vector>

#include "exceptions.hpp"
#include "hessianInputParser.hpp"
#include "hessianSettings.hpp"
#include "testInputFileReader.hpp"
#include "throwWithMessage.hpp"

using namespace input;
using namespace settings;

TEST_F(TestInputFileReader, parseHessianFile)
{
    HessianInputParser      parser(*_engine);
    std::vector<std::string> lineElements = {
        "hessian_file",
        "=",
        "water.hessian"
    };

    parser.parseHessianFile(lineElements, 0);

    EXPECT_EQ(HessianSettings::getHessianFile(), "water.hessian");
}

TEST_F(TestInputFileReader, parseHessianDisplacement)
{
    HessianInputParser      parser(*_engine);
    std::vector<std::string> lineElements = {
        "hessian_displacement",
        "=",
        "0.001"
    };

    parser.parseDisplacement(lineElements, 0);

    EXPECT_EQ(HessianSettings::getDisplacement(), 0.001);

    lineElements = {"hessian_displacement", "=", "0.0"};

    EXPECT_THROW_MSG(
        parser.parseDisplacement(lineElements, 7),
        customException::InputFileException,
        "Hessian displacement must be greater than 0 in input file at line 7"
    );
}

TEST_F(TestInputFileReader, parseHessianBuilder)
{
    HessianInputParser      parser(*_engine);
    std::vector<std::string> lineElements = {
        "hessian_builder",
        "=",
        "five-point"
    };

    parser.parseBuilder(lineElements, 0);

    EXPECT_EQ(
        HessianSettings::getBuilder(),
        HessianBuilderType::FINITE_DIFFERENCE_FORCES_FIVE_POINT
    );

    lineElements = {"hessian_builder", "=", "unknown"};

    EXPECT_THROW_MSG(
        parser.parseBuilder(lineElements, 9),
        customException::InputFileException,
        "Invalid hessian_builder \"unknown\" in input file at line 9 - "
        "possible values are: central, forward, five-point, analytic"
    );
}

TEST_F(TestInputFileReader, parseOptimizeBeforeHessian)
{
    HessianInputParser       parser(*_engine);
    std::vector<std::string> lineElements = {
        "optimize_before_hessian",
        "=",
        "off"
    };

    parser.parseOptimizeBeforeHessian(lineElements, 0);
    EXPECT_FALSE(HessianSettings::optimizeBeforeHessian());

    lineElements = {"optimize_before_hessian", "=", "on"};

    parser.parseOptimizeBeforeHessian(lineElements, 0);
    EXPECT_TRUE(HessianSettings::optimizeBeforeHessian());
}
