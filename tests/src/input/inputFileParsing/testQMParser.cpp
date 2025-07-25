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

    parser.parseQMMethod({"qm_prog", "=", "ase_dftbplus"}, 0);
    EXPECT_EQ(QMSettings::getQMMethod(), ASEDFTBPLUS);

    parser.parseQMMethod({"qm_prog", "=", "ase_xtb"}, 0);
    EXPECT_EQ(QMSettings::getQMMethod(), ASEXTB);

    // the more detailed mace parser is tested in TestMaceParser

    ASSERT_THROW_MSG(
        parser.parseQMMethod({"qm_prog", "=", "notAMethod"}, 0),
        InputFileException,
        "Invalid qm_prog \"notAMethod\" in input file.\n"
        "Possible values are: dftbplus, ase_dftbplus, ase_xtb, pyscf, "
        "turbomole, mace, mace_mp, mace_off"
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
        {"qm_script_full_path", "=", "/path/to/QM/Script.sh"},
        0
    );
    EXPECT_EQ(QMSettings::getQMScriptFullPath(), "/path/to/QM/Script.sh");
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
    EXPECT_FALSE(QMSettings::useDispersionCorr());

    auto parser = QMInputParser(*_engine);
    parser.parseDispersion({"dispersion", "=", "true"}, 0);
    EXPECT_TRUE(QMSettings::useDispersionCorr());

    parser.parseDispersion({"dispersion", "=", "yes"}, 0);
    EXPECT_TRUE(QMSettings::useDispersionCorr());

    parser.parseDispersion({"dispersion", "=", "on"}, 0);
    EXPECT_TRUE(QMSettings::useDispersionCorr());

    parser.parseDispersion({"dispersion", "=", "false"}, 0);
    EXPECT_FALSE(QMSettings::useDispersionCorr());

    parser.parseDispersion({"dispersion", "=", "no"}, 0);
    EXPECT_FALSE(QMSettings::useDispersionCorr());

    parser.parseDispersion({"dispersion", "=", "off"}, 0);
    EXPECT_FALSE(QMSettings::useDispersionCorr());

    ASSERT_THROW_MSG(
        parser.parseDispersion({"dispersion", "=", "notABool"}, 0),
        InputFileException,
        "Invalid boolean option \"notABool\" for keyword \"dispersion\" in "
        "input file.\n"
        "Possible values are: on, yes, true, off, no, false."
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

    parser.parseMaceModelSize({"mace_model_size", "=", "small_0b"}, 0);
    EXPECT_EQ(QMSettings::getMaceModelSize(), SMALL0B);

    parser.parseMaceModelSize({"mace_model_size", "=", "medium_0b"}, 0);
    EXPECT_EQ(QMSettings::getMaceModelSize(), MEDIUM0B);

    parser.parseMaceModelSize({"mace_model_size", "=", "small_0b2"}, 0);
    EXPECT_EQ(QMSettings::getMaceModelSize(), SMALL0B2);

    parser.parseMaceModelSize({"mace_model_size", "=", "medium_0b2"}, 0);
    EXPECT_EQ(QMSettings::getMaceModelSize(), MEDIUM0B2);

    parser.parseMaceModelSize({"mace_model_size", "=", "large_0b2"}, 0);
    EXPECT_EQ(QMSettings::getMaceModelSize(), LARGE0B2);

    parser.parseMaceModelSize({"mace_model_size", "=", "medium_0b3"}, 0);
    EXPECT_EQ(QMSettings::getMaceModelSize(), MEDIUM0B3);

    parser.parseMaceModelSize({"mace_model_size", "=", "medium_mpa_0"}, 0);
    EXPECT_EQ(QMSettings::getMaceModelSize(), MEDIUMMPA0);

    parser.parseMaceModelSize({"mace_model_size", "=", "medium_omat_0"}, 0);
    EXPECT_EQ(QMSettings::getMaceModelSize(), MEDIUMOMAT0);

    parser.parseMaceModelSize({"mace_model_size", "=", "custom"}, 0);
    EXPECT_EQ(QMSettings::getMaceModelSize(), CUSTOM);

    ASSERT_THROW_MSG(
        parser.parseMaceModelSize({"mace_model_size", "=", "notASize"}, 0),
        InputFileException,
        "Invalid mace_model_size \"notASize\" in input file.\n"
        "Possible values are: small, medium, large, small-0b,\n"
        "medium-0b, small-0b2, medium-0b2, large-0b2, medium-0b3,\n"
        "medium-mpa-0, medium-omat-0, custom"
    )
}

TEST_F(TestInputFileReader, parseSlakosType)
{
    using enum QMMethod;

    auto parser = QMInputParser(*_engine);

    parser.parseSlakosType({"slakos", "=", "3ob"}, 0);
    EXPECT_EQ(QMSettings::getSlakosType(), SlakosType::THREEOB);

    parser.parseSlakosType({"slakos", "=", "matsci"}, 0);
    EXPECT_EQ(QMSettings::getSlakosType(), SlakosType::MATSCI);

    parser.parseSlakosType({"slakos", "=", "custom"}, 0);
    EXPECT_EQ(QMSettings::getSlakosType(), SlakosType::CUSTOM);

    ASSERT_THROW_MSG(
        parser.parseSlakosType({"slakos", "=", "notASlakosType"}, 0),
        InputFileException,
        "Invalid slakos type \"notASlakosType\" in input file.\n"
        "Possible values are: 3ob, matsci, custom"
    )
}

TEST_F(TestInputFileReader, parseSlakosTypeThirdOrder)
{
    using enum QMMethod;

    auto parser1 = QMInputParser(*_engine);

    parser1.parseThirdOrder({"third_order", "=", "off"}, 0);
    parser1.parseSlakosType({"slakos", "=", "3ob"}, 0);
    EXPECT_EQ(QMSettings::getSlakosType(), SlakosType::THREEOB);
    EXPECT_FALSE(QMSettings::useThirdOrderDftb());

    auto parser2 = QMInputParser(*_engine);
    parser2.parseSlakosType({"slakos", "=", "3ob"}, 0);
    parser2.parseThirdOrder({"third_order", "=", "off"}, 0);
    EXPECT_EQ(QMSettings::getSlakosType(), SlakosType::THREEOB);
    EXPECT_FALSE(QMSettings::useThirdOrderDftb());
}

TEST_F(TestInputFileReader, parseSlakosPath)
{
    using enum QMMethod;

    auto parser = QMInputParser(*_engine);
    parser.parseSlakosType({"slakos", "=", "custom"}, 0);
    parser.parseSlakosPath({"slakos_path", "=", "/path/to/slakos"}, 0);
    EXPECT_EQ(QMSettings::getSlakosPath(), "/path/to/slakos");
}

TEST_F(TestInputFileReader, parseThirdOrder)
{
    EXPECT_FALSE(QMSettings::useThirdOrderDftb());

    auto parser = QMInputParser(*_engine);
    parser.parseThirdOrder({"third_order", "=", "on"}, 0);
    EXPECT_TRUE(QMSettings::useThirdOrderDftb());

    parser.parseThirdOrder({"third_order", "=", "off"}, 0);
    EXPECT_FALSE(QMSettings::useThirdOrderDftb());

    parser.parseThirdOrder({"third_order", "=", "true"}, 0);
    EXPECT_TRUE(QMSettings::useThirdOrderDftb());

    parser.parseThirdOrder({"third_order", "=", "false"}, 0);
    EXPECT_FALSE(QMSettings::useThirdOrderDftb());

    parser.parseThirdOrder({"third_order", "=", "yes"}, 0);
    EXPECT_TRUE(QMSettings::useThirdOrderDftb());

    parser.parseThirdOrder({"third_order", "=", "no"}, 0);
    EXPECT_FALSE(QMSettings::useThirdOrderDftb());
    EXPECT_TRUE(QMSettings::isThirdOrderDftbSet());

    ASSERT_THROW_MSG(
        parser.parseThirdOrder({"third_order", "=", "notABool"}, 0),
        InputFileException,
        "Invalid boolean option \"notABool\" for keyword \"third_order\" in "
        "input file.\n"
        "Possible values are: on, yes, true, off, no, false."
    )
}

TEST_F(TestInputFileReader, parseHubbardDerivs)
{
    auto parser = QMInputParser(*_engine);

    parser.parseHubbardDerivs({"hubbard_derivs", "=", "H:1.0,He:2.0"}, 0);

    const auto hubbardDerivs = QMSettings::getHubbardDerivs();
    EXPECT_EQ(hubbardDerivs.size(), 2);
    EXPECT_EQ(hubbardDerivs.at("H"), 1.0);
    EXPECT_EQ(hubbardDerivs.at("He"), 2.0);

    ASSERT_THROW_MSG(
        parser.parseHubbardDerivs({"hubbard_derivs", "=", "H:1.0,He"}, 0),
        InputFileException,
        "Invalid hubbard_derivs format \"H:1.0,He\" in input file."
    )
}

TEST_F(TestInputFileReader, parseXtbMethod)
{
    using enum QMMethod;

    auto parser = QMInputParser(*_engine);

    parser.parseXtbMethod({"xtb_method", "=", "Gfn1-XTb"}, 0);
    EXPECT_EQ(QMSettings::getXtbMethod(), XtbMethod::GFN1);

    parser.parseXtbMethod({"xtb_method", "=", "gfN2-XTb"}, 0);
    EXPECT_EQ(QMSettings::getXtbMethod(), XtbMethod::GFN2);

    parser.parseXtbMethod({"xtb_method", "=", "iPEa1-XTb"}, 0);
    EXPECT_EQ(QMSettings::getXtbMethod(), XtbMethod::IPEA1);

    ASSERT_THROW_MSG(
        parser.parseXtbMethod({"xtb_method", "=", "notAnXtbMethod"}, 0),
        InputFileException,
        "Invalid xTB method \"notAnXtbMethod\" in input file.\n"
        "Possible values are: GFN1-xTB, GFN2-xTB, IPEA1-xTB"
    )
}