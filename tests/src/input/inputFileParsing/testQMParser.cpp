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

#include <gtest/gtest.h>   // for TEST_F, EXPECT_EQ, RUN_ALL_TESTS

#include <string>   // for string, allocator

#include "QMInputParser.hpp"         // for InputFileParserQM
#include "exceptions.hpp"            // for InputFileException, customException
#include "gtest/gtest.h"             // for Message, TestPartResult
#include "inputFileParser.hpp"       // for readInput
#include "qmSettings.hpp"            // for QMSettings
#include "testInputFileReader.hpp"   // for TestInputFileReader
#include "throwWithMessage.hpp"      // for ASSERT_THROW_MSG

using namespace input;
using namespace settings;
using namespace customException;

TEST_F(TestInputFileReader, parseQMMethod)
{
    using enum QMMethod;
    EXPECT_EQ(QMSettings::getQMMethod(), NONE);

    auto parser = QMInputParser(*_engine);
    parser.parseQMMethod({"qm_prog", "=", "dftbplus"}, 0);
    EXPECT_EQ(QMSettings::getQMMethod(), DFTBPLUS);

    parser.parseQMMethod({"qm_prog", "=", "pyscf"}, 0);
    EXPECT_EQ(QMSettings::getQMMethod(), PYSCF);

    parser.parseQMMethod({"qm_prog", "=", "turbomole"}, 0);
    EXPECT_EQ(QMSettings::getQMMethod(), TURBOMOLE);

    parser.parseQMMethod({"qm_prog", "=", "mace"}, 0);
    EXPECT_EQ(QMSettings::getQMMethod(), MACE);

    // the more detailed mace parser is tested in TestMaceParser

    ASSERT_THROW_MSG(
        parser.parseQMMethod({"qm_prog", "=", "notAMethod"}, 0),
        InputFileException,
        "Invalid qm_prog \"notAMethod\" in input file.\n"
        "Possible values are: dftbplus, pyscf, turbomole, mace, mace_mp, "
        "mace_off"
    )
}

TEST_F(TestInputFileReader, parseMaceQMMethod)
{
    using enum QMMethod;
    using enum MaceModelType;

    auto parser = QMInputParser(*_engine);

    parser.parseMaceQMMethod("mace");
    EXPECT_EQ(QMSettings::getQMMethod(), MACE);
    EXPECT_EQ(QMSettings::getMaceModelType(), MACE_MP);

    parser.parseMaceQMMethod("mace_mp");
    EXPECT_EQ(QMSettings::getQMMethod(), MACE);
    EXPECT_EQ(QMSettings::getMaceModelType(), MACE_MP);

    parser.parseMaceQMMethod("mace_off");
    EXPECT_EQ(QMSettings::getQMMethod(), MACE);
    EXPECT_EQ(QMSettings::getMaceModelType(), MACE_OFF);

    ASSERT_THROW_MSG(
        parser.parseMaceQMMethod("mace_ani"),
        InputFileException,
        "The mace ani model is not supported in this version of PQ.\n"
    )

    ASSERT_THROW_MSG(
        parser.parseMaceQMMethod("mace_anicc"),
        InputFileException,
        "The mace ani model is not supported in this version of PQ.\n"
    )

    ASSERT_THROW_MSG(
        parser.parseMaceQMMethod("notAMaceModel"),
        InputFileException,
        "Invalid mace type qm_method \"notAMaceModel\" in input file.\n"
        "Possible values are: mace (mace_mp), mace_off"
    )
}

TEST_F(TestInputFileReader, parseQMScript)
{
    auto parser = QMInputParser(*_engine);
    parser.parseQMScript({"qm_script", "=", "script.sh"}, 0);
    EXPECT_EQ(QMSettings::getQMScript(), "script.sh");
}

TEST_F(TestInputFileReader, parseQMScriptFullPath)
{
    auto parser = QMInputParser(*_engine);
    parser.parseQMScriptFullPath(
        {"qm_script_full_path", "=", "/path/to/script.sh"},
        0
    );
    EXPECT_EQ(QMSettings::getQMScriptFullPath(), "/path/to/script.sh");
}

TEST_F(TestInputFileReader, parseQMLoopTimeLimit)
{
    auto parser = QMInputParser(*_engine);
    parser.parseQMLoopTimeLimit({"qm_loop_time_limit", "=", "10"}, 0);
    EXPECT_EQ(QMSettings::getQMLoopTimeLimit(), 10);

    parser.parseQMLoopTimeLimit({"qm_loop_time_limit", "=", "-1"}, 0);
    EXPECT_EQ(QMSettings::getQMLoopTimeLimit(), -1);
}

TEST_F(TestInputFileReader, parseDispersion)
{
    EXPECT_FALSE(QMSettings::useDispersionCorrection());

    auto parser = QMInputParser(*_engine);
    parser.parseDispersion({"dispersion", "=", "on"}, 0);
    EXPECT_TRUE(QMSettings::useDispersionCorrection());

    parser.parseDispersion({"dispersion", "=", "off"}, 0);
    EXPECT_FALSE(QMSettings::useDispersionCorrection());

    parser.parseDispersion({"dispersion", "=", "true"}, 0);
    EXPECT_TRUE(QMSettings::useDispersionCorrection());

    parser.parseDispersion({"dispersion", "=", "false"}, 0);
    EXPECT_FALSE(QMSettings::useDispersionCorrection());

    ASSERT_THROW_MSG(
        parser.parseDispersion({"dispersion", "=", "notABool"}, 0),
        InputFileException,
        "Invalid dispersion \"notABool\" in input file.\n"
        "Possible values are: true, false, on, off"
    )
}

TEST_F(TestInputFileReader, parseMaceModelSize)
{
    using enum MaceModelSize;

    auto parser = QMInputParser(*_engine);
    parser.parseMaceModelSize({"mace_model_size", "=", "small"}, 0);
    EXPECT_EQ(QMSettings::getMaceModelSize(), SMALL);

    parser.parseMaceModelSize({"mace_model_size", "=", "medium"}, 0);
    EXPECT_EQ(QMSettings::getMaceModelSize(), MEDIUM);

    parser.parseMaceModelSize({"mace_model_size", "=", "large"}, 0);
    EXPECT_EQ(QMSettings::getMaceModelSize(), LARGE);

    ASSERT_THROW_MSG(
        parser.parseMaceModelSize({"mace_model_size", "=", "notASize"}, 0),
        InputFileException,
        "Invalid mace_model_size \"notASize\" in input file.\n"
        "Possible values are: small, medium, large"
    )
}