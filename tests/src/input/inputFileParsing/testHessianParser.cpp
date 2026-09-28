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

TEST_F(TestInputFileReader, parseHessianFile)
{
    input::HessianInputParser parser;
    const auto                funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("hessian_file"));
    const auto& parseFunc = funcMap.at("hessian_file");

    std::vector<std::string> lineElements = {
        "hessian_file",
        "=",
        "water.hessian"
    };
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::HessianSettings::getHessianFile(), "water.hessian");
}

TEST_F(TestInputFileReader, parseHessianInfoFile)
{
    input::HessianInputParser parser;
    const auto                funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("hessian_info_file"));
    const auto& parseFunc = funcMap.at("hessian_info_file");

    std::vector<std::string> lineElements = {
        "hessian_info_file",
        "=",
        "water.hessian.info"
    };
    parseFunc(lineElements, 0);
    EXPECT_EQ(
        settings::HessianSettings::getHessianInfoFile(),
        "water.hessian.info"
    );
}

TEST_F(TestInputFileReader, parseHessianDisplacement)
{
    input::HessianInputParser parser;
    const auto                funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("hessian_displacement"));
    const auto& parseFunc = funcMap.at("hessian_displacement");

    std::vector<std::string> lineElements = {
        "hessian_displacement",
        "=",
        "0.001"
    };
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::HessianSettings::getDisplacement(), 0.001);

    clearParser(parser);

    lineElements = {"hessian_displacement", "=", "0.0"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 7),
        exc::InputFileException,
        "Invalid value \"0.0\" for key \"hessian_displacement\" at line 7 in "
        "input file: failed validation with message Value must be greater than "
        "0"
    );
}

TEST_F(TestInputFileReader, parseHessianBuilder)
{
    input::HessianInputParser parser;
    const auto                funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("hessian_builder"));
    const auto& parseFunc = funcMap.at("hessian_builder");

    std::vector<std::string> lineElements = {
        "hessian_builder",
        "=",
        "five-point"
    };
    parseFunc(lineElements, 0);
    EXPECT_EQ(
        settings::HessianSettings::getBuilder(),
        HessianBuilderType::FIVE_POINT
    );

    clearParser(parser);

    lineElements = {"hessian_builder", "=", "unknown"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 9),
        exc::InputFileException,
        "Invalid value \"unknown\" for key \"hessian_builder\" at line 9 in "
        "input file. Allowed values: central, forward, five_point, analytic"
    );
}

TEST_F(TestInputFileReader, parseOptimizeBeforeHessian)
{
    input::HessianInputParser parser;
    const auto                funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("optimize_before_hessian"));
    const auto& parseFunc = funcMap.at("optimize_before_hessian");

    std::vector<std::string> lineElements = {
        "optimize_before_hessian",
        "=",
        "off"
    };
    parseFunc(lineElements, 0);
    EXPECT_FALSE(settings::HessianSettings::optimizeBeforeHessian());

    clearParser(parser);

    lineElements = {"optimize_before_hessian", "=", "on"};
    parseFunc(lineElements, 0);
    EXPECT_TRUE(settings::HessianSettings::optimizeBeforeHessian());
}
