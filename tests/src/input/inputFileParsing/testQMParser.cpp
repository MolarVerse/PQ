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

#include "QMInputParser.hpp"   // for InputFileParserQM
#include "exceptions.hpp"      // for exc::InputFileException, customException
#include "qmSettings.hpp"      // for settings::QMSettings
#include "testInputFileReader.hpp"   // for TestInputFileReader
#include "throwWithMessage.hpp"      // for ASSERT_THROW_MSG

TEST_F(TestInputFileReader, parseQMMethod)
{
    using enum QMMethod;
    EXPECT_EQ(settings::QMSettings::getQMMethod(), NONE);

    auto       parser  = input::QMInputParser();
    const auto funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("qm_prog"));
    const auto& parseFunc = funcMap.at("qm_prog");

    parseFunc({"qm_prog", "=", "dftbplus"}, 0);
    EXPECT_EQ(settings::QMSettings::getQMMethod(), DFTBPLUS);

    _clearParser(parser);

    parseFunc({"qm_prog", "=", "pyscf"}, 0);
    EXPECT_EQ(settings::QMSettings::getQMMethod(), PYSCF);

    _clearParser(parser);

    parseFunc({"qm_prog", "=", "turbomole"}, 0);
    EXPECT_EQ(settings::QMSettings::getQMMethod(), TURBOMOLE);

    _clearParser(parser);

    parseFunc({"qm_prog", "=", "mace"}, 0);
    EXPECT_EQ(settings::QMSettings::getQMMethod(), MACE);

    _clearParser(parser);

    parseFunc({"qm_prog", "=", "ase_dftbplus"}, 0);
    EXPECT_EQ(settings::QMSettings::getQMMethod(), ASE_DFTBPLUS);

    _clearParser(parser);

    parseFunc({"qm_prog", "=", "ase_xtb"}, 0);
    EXPECT_EQ(settings::QMSettings::getQMMethod(), ASE_XTB);

    _clearParser(parser);

    parseFunc({"qm_prog", "=", "fennol"}, 0);
    EXPECT_EQ(settings::QMSettings::getQMMethod(), FENNOL);

    _clearParser(parser);

    parseFunc({"qm_prog", "=", "mace"}, 0);
    EXPECT_EQ(settings::QMSettings::getQMMethod(), MACE);

    _clearParser(parser);

    parseFunc({"qm_prog", "=", "mace_mp"}, 0);
    EXPECT_EQ(settings::QMSettings::getQMMethod(), MACE);

    _clearParser(parser);

    parseFunc({"qm_prog", "=", "mace_off"}, 0);
    EXPECT_EQ(settings::QMSettings::getQMMethod(), MACE);

    _clearParser(parser);

    EXPECT_THROW_MSG(
        parseFunc({"qm_prog", "=", "mace-ani"}, 0),
        exc::InputFileException,
        "The mace ani model is not supported in this version of PQ.\n"
    );

    _clearParser(parser);

    EXPECT_THROW_MSG(
        parseFunc({"qm_prog", "=", "mace-anicc"}, 0),
        exc::InputFileException,
        "The mace ani model is not supported in this version of PQ.\n"
    );

    _clearParser(parser);

    ASSERT_THROW_MSG(
        parseFunc({"qm_prog", "=", "notAMethod"}, 0),
        exc::InputFileException,
        "Invalid value \"notAMethod\" for key \"qm_prog\" at line 0 in input "
        "file. Allowed values: dftbplus, ase_dftbplus, ase_xtb, pyscf, "
        "turbomole, mace, fennol, mace_mp, mace_off, mace_anicc, mace_ani"
    )
}

TEST_F(TestInputFileReader, parseQMScript)
{
    auto       parser  = input::QMInputParser();
    const auto funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("qm_script"));
    const auto& parseFunc = funcMap.at("qm_script");
    parseFunc({"qm_script", "=", "script.sh"}, 0);
    EXPECT_EQ(settings::QMSettings::getQMScript(), "script.sh");
}

TEST_F(TestInputFileReader, parseQMScriptFullPath)
{
    auto       parser  = input::QMInputParser();
    const auto funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("qm_script_full_path"));
    const auto& parseFunc = funcMap.at("qm_script_full_path");
    parseFunc({"qm_script_full_path", "=", "/path/to/QM/Script.sh"}, 0);
    EXPECT_EQ(
        settings::QMSettings::getQMScriptFullPath(),
        "/path/to/QM/Script.sh"
    );
}

TEST_F(TestInputFileReader, parseQMLoopTimeLimit)
{
    auto       parser  = input::QMInputParser();
    const auto funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("qm_loop_time_limit"));
    const auto& parseFunc = funcMap.at("qm_loop_time_limit");
    parseFunc({"qm_loop_time_limit", "=", "10"}, 0);
    EXPECT_EQ(settings::QMSettings::getQMLoopTimeLimit(), 10);

    _clearParser(parser);

    parseFunc({"qm_loop_time_limit", "=", "-1"}, 0);
    EXPECT_EQ(settings::QMSettings::getQMLoopTimeLimit(), -1);
}

TEST_F(TestInputFileReader, parseDispersion)
{
    EXPECT_FALSE(settings::QMSettings::useDispersionCorr());

    auto       parser  = input::QMInputParser();
    const auto funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("dispersion"));
    const auto& parseFunc = funcMap.at("dispersion");

    parseFunc({"dispersion", "=", "true"}, 0);
    EXPECT_TRUE(settings::QMSettings::useDispersionCorr());

    _clearParser(parser);

    parseFunc({"dispersion", "=", "yes"}, 0);
    EXPECT_TRUE(settings::QMSettings::useDispersionCorr());

    _clearParser(parser);

    parseFunc({"dispersion", "=", "on"}, 0);
    EXPECT_TRUE(settings::QMSettings::useDispersionCorr());

    _clearParser(parser);

    parseFunc({"dispersion", "=", "false"}, 0);
    EXPECT_FALSE(settings::QMSettings::useDispersionCorr());

    _clearParser(parser);

    parseFunc({"dispersion", "=", "no"}, 0);
    EXPECT_FALSE(settings::QMSettings::useDispersionCorr());

    _clearParser(parser);

    parseFunc({"dispersion", "=", "off"}, 0);
    EXPECT_FALSE(settings::QMSettings::useDispersionCorr());

    _clearParser(parser);

    ASSERT_THROW_MSG(
        parseFunc({"dispersion", "=", "notABool"}, 0),
        exc::InputFileException,
        "Invalid value \"notABool\" for key \"dispersion\" at line 0 in input "
        "file. Allowed values: on|off|true|false|yes|no"
    )
}

TEST_F(TestInputFileReader, parseRemoveNetForce)
{
    EXPECT_FALSE(settings::QMSettings::getRemoveNetForce());

    auto       parser  = input::QMInputParser();
    const auto funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("remove_net_force"));
    const auto& parseFunc = funcMap.at("remove_net_force");

    parseFunc({"remove_net_force", "=", "true"}, 0);
    EXPECT_TRUE(settings::QMSettings::getRemoveNetForce());

    _clearParser(parser);

    parseFunc({"remove_net_force", "=", "yes"}, 0);
    EXPECT_TRUE(settings::QMSettings::getRemoveNetForce());

    _clearParser(parser);

    parseFunc({"remove_net_force", "=", "on"}, 0);
    EXPECT_TRUE(settings::QMSettings::getRemoveNetForce());

    _clearParser(parser);

    parseFunc({"remove_net_force", "=", "false"}, 0);
    EXPECT_FALSE(settings::QMSettings::getRemoveNetForce());

    _clearParser(parser);

    parseFunc({"remove_net_force", "=", "no"}, 0);
    EXPECT_FALSE(settings::QMSettings::getRemoveNetForce());

    _clearParser(parser);

    parseFunc({"remove_net_force", "=", "off"}, 0);
    EXPECT_FALSE(settings::QMSettings::getRemoveNetForce());

    _clearParser(parser);

    ASSERT_THROW_MSG(
        parseFunc({"remove_net_force", "=", "notABool"}, 0),
        exc::InputFileException,
        "Invalid value \"notABool\" for key \"remove_net_force\" at line 0 in "
        "input file. Allowed values: on|off|true|false|yes|no"
    );
}

TEST_F(TestInputFileReader, parseMaceQMMethod)
{
    using enum QMMethod;
    using enum MaceModelType;

    auto parser = input::QMInputParser();

    input::QMInputParser::parseMaceQMMethod("mace");
    EXPECT_EQ(settings::QMSettings::getMaceModelType(), MACE_MP);

    input::QMInputParser::parseMaceQMMethod("mace_mp");
    EXPECT_EQ(settings::QMSettings::getMaceModelType(), MACE_MP);

    input::QMInputParser::parseMaceQMMethod("mace_off");
    EXPECT_EQ(settings::QMSettings::getMaceModelType(), MACE_OFF);

    ASSERT_THROW_MSG(
        parser.parseMaceQMMethod("mace_ani"),
        exc::InputFileException,
        "The mace ani model is not supported in this version of PQ.\n"
    )

    ASSERT_THROW_MSG(
        parser.parseMaceQMMethod("mace_anicc"),
        exc::InputFileException,
        "The mace ani model is not supported in this version of PQ.\n"
    )

    ASSERT_THROW_MSG(
        parser.parseMaceQMMethod("notAMaceModel"),
        exc::InputFileException,
        "Invalid mace type qm_method \"notAMaceModel\" in input file.\n"
        "Possible values are: mace_mp, mace_off, mace_anicc, mace, mace_ani"
    )
}

TEST_F(TestInputFileReader, parseMaceModel)
{
    using enum MaceModel;

    auto       parser  = input::QMInputParser();
    const auto funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("mace_model"));
    const auto& parseFunc = funcMap.at("mace_model");

    parseFunc({"mace_model", "=", "small"}, 0);
    EXPECT_EQ(settings::QMSettings::getMaceModel(), SMALL);

    _clearParser(parser);

    parseFunc({"mace_model", "=", "medium"}, 0);
    EXPECT_EQ(settings::QMSettings::getMaceModel(), MEDIUM);

    _clearParser(parser);

    parseFunc({"mace_model", "=", "large"}, 0);
    EXPECT_EQ(settings::QMSettings::getMaceModel(), LARGE);

    _clearParser(parser);

    parseFunc({"mace_model", "=", "small_0b"}, 0);
    EXPECT_EQ(settings::QMSettings::getMaceModel(), SMALL_0B);

    _clearParser(parser);

    parseFunc({"mace_model", "=", "medium_0b"}, 0);
    EXPECT_EQ(settings::QMSettings::getMaceModel(), MEDIUM_0B);

    _clearParser(parser);

    parseFunc({"mace_model", "=", "small_0b2"}, 0);
    EXPECT_EQ(settings::QMSettings::getMaceModel(), SMALL_0B2);

    _clearParser(parser);

    parseFunc({"mace_model", "=", "medium_0b2"}, 0);
    EXPECT_EQ(settings::QMSettings::getMaceModel(), MEDIUM_0B2);

    _clearParser(parser);

    parseFunc({"mace_model", "=", "large_0b2"}, 0);
    EXPECT_EQ(settings::QMSettings::getMaceModel(), LARGE_0B2);

    _clearParser(parser);

    parseFunc({"mace_model", "=", "medium_0b3"}, 0);
    EXPECT_EQ(settings::QMSettings::getMaceModel(), MEDIUM_0B3);

    _clearParser(parser);

    parseFunc({"mace_model", "=", "medium_mpa_0"}, 0);
    EXPECT_EQ(settings::QMSettings::getMaceModel(), MEDIUM_MPA_0);

    _clearParser(parser);

    parseFunc({"mace_model", "=", "medium_omat_0"}, 0);
    EXPECT_EQ(settings::QMSettings::getMaceModel(), MEDIUM_OMAT_0);

    _clearParser(parser);

    parseFunc({"mace_model", "=", "custom"}, 0);
    EXPECT_EQ(settings::QMSettings::getMaceModel(), CUSTOM);

    _clearParser(parser);

    ASSERT_THROW_MSG(
        parseFunc({"mace_model", "=", "notASize"}, 0),
        exc::InputFileException,
        "Invalid value \"notASize\" for key \"mace_model\" at line 0 in input "
        "file. Allowed values: small, medium, large, small_0b, medium_0b, "
        "small_0b2, medium_0b2, large_0b2, medium_0b3, medium_mpa_0, "
        "medium_omat_0, custom"
    )
}

TEST_F(TestInputFileReader, parseMaceMode)
{
    using enum MaceMode;

    auto       parser  = input::QMInputParser();
    const auto funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("mace_mode"));
    const auto& parseFunc = funcMap.at("mace_mode");

    parseFunc({"mace_mode", "=", "accurate"}, 0);
    EXPECT_EQ(settings::QMSettings::getMaceMode(), ACCURATE);

    _clearParser(parser);

    parseFunc({"mace_mode", "=", "fast"}, 0);
    EXPECT_EQ(settings::QMSettings::getMaceMode(), FAST);

    _clearParser(parser);

    ASSERT_THROW_MSG(
        parseFunc({"mace_mode", "=", "notAMode"}, 0),
        exc::InputFileException,
        "Invalid value \"notAMode\" for key \"mace_mode\" at line 0 in input "
        "file. Allowed values: accurate, fast"
    )
}

TEST_F(TestInputFileReader, parseMaceModelPath)
{
    auto       parser  = input::QMInputParser();
    const auto funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("mace_model_path"));
    const auto& parseFunc = funcMap.at("mace_model_path");

    settings::QMSettings::setMaceModelPath("");
    EXPECT_EQ(settings::QMSettings::getMaceModelPath(), "");
    parseFunc({"mace_model_path", "=", "/pAth/to/mace.model"}, 0);
    EXPECT_EQ(settings::QMSettings::getMaceModelPath(), "/pAth/to/mace.model");
}

TEST_F(TestInputFileReader, parseSlakosType)
{
    using enum QMMethod;

    auto       parser  = input::QMInputParser();
    const auto funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("slakos"));
    const auto& parseFunc = funcMap.at("slakos");

#ifdef WITH_ASE
    parseFunc({"slakos", "=", "3ob"}, 0);
    EXPECT_EQ(settings::QMSettings::getSlakosType(), SlakosType::THREEOB);

    _clearParser(parser);

    parseFunc({"slakos", "=", "matsci"}, 0);
    EXPECT_EQ(settings::QMSettings::getSlakosType(), SlakosType::MATSCI);

    _clearParser(parser);
#else
    ASSERT_THROW_MSG(
        parseFunc({"slakos", "=", "3ob"}, 0),
        exc::InputFileException,
        "Built-in SLAKOS sets (3ob/matsci) require building PQ with "
        "-DBUILD_WITH_ASE=On"
    );

    _clearParser(parser);

    ASSERT_THROW_MSG(
        parseFunc({"slakos", "=", "matsci"}, 0),
        exc::InputFileException,
        "Built-in SLAKOS sets (3ob/matsci) require building PQ with "
        "-DBUILD_WITH_ASE=On"
    );

    _clearParser(parser);
#endif

    parseFunc({"slakos", "=", "custom"}, 0);
    EXPECT_EQ(settings::QMSettings::getSlakosType(), SlakosType::CUSTOM);

    _clearParser(parser);

    ASSERT_THROW_MSG(
        parseFunc({"slakos", "=", "notASlakosType"}, 0),
        exc::InputFileException,
        "Invalid value \"notASlakosType\" for key \"slakos\" at line 0 in "
        "input file. Allowed values: threeob, matsci, custom, 3ob"
    )
}

#ifdef WITH_ASE
TEST_F(TestInputFileReader, parseSlakosTypeThirdOrder)
{
    using enum QMMethod;

    auto       parser  = input::QMInputParser();
    const auto funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("slakos"));
    const auto& slakosParseFunc = funcMap.at("slakos");
    ASSERT_TRUE(funcMap.contains("third_order"));
    const auto& thirdOrderParseFunc = funcMap.at("third_order");

    thirdOrderParseFunc({"third_order", "=", "off"}, 0);
    slakosParseFunc({"slakos", "=", "3ob"}, 0);
    EXPECT_EQ(settings::QMSettings::getSlakosType(), SlakosType::THREEOB);
    EXPECT_FALSE(settings::QMSettings::useThirdOrderDftb());

    _clearParser(parser);

    auto parser2 = input::QMInputParser();
    slakosParseFunc({"slakos", "=", "3ob"}, 0);
    thirdOrderParseFunc({"third_order", "=", "off"}, 0);
    EXPECT_EQ(settings::QMSettings::getSlakosType(), SlakosType::THREEOB);
    EXPECT_FALSE(settings::QMSettings::useThirdOrderDftb());
}
#endif

TEST_F(TestInputFileReader, parseSlakosPath)
{
    using enum QMMethod;

    auto       parser  = input::QMInputParser();
    const auto funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("slakos"));
    const auto& slakosParseFunc = funcMap.at("slakos");
    ASSERT_TRUE(funcMap.contains("slakos_path"));
    const auto& slakosPathParseFunc = funcMap.at("slakos_path");

    slakosParseFunc({"slakos", "=", "custom"}, 0);
    slakosPathParseFunc({"slakos_path", "=", "/path/to/slakos"}, 0);
    EXPECT_EQ(settings::QMSettings::getSlakosPath(), "/path/to/slakos");
}

TEST_F(TestInputFileReader, parseThirdOrder)
{
    EXPECT_FALSE(settings::QMSettings::useThirdOrderDftb());

    auto       parser  = input::QMInputParser();
    const auto funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("third_order"));
    const auto& thirdOrderParseFunc = funcMap.at("third_order");

    thirdOrderParseFunc({"third_order", "=", "on"}, 0);
    EXPECT_TRUE(settings::QMSettings::useThirdOrderDftb());

    _clearParser(parser);

    thirdOrderParseFunc({"third_order", "=", "off"}, 0);
    EXPECT_FALSE(settings::QMSettings::useThirdOrderDftb());

    _clearParser(parser);

    thirdOrderParseFunc({"third_order", "=", "true"}, 0);
    EXPECT_TRUE(settings::QMSettings::useThirdOrderDftb());

    _clearParser(parser);

    thirdOrderParseFunc({"third_order", "=", "false"}, 0);
    EXPECT_FALSE(settings::QMSettings::useThirdOrderDftb());

    _clearParser(parser);

    thirdOrderParseFunc({"third_order", "=", "yes"}, 0);
    EXPECT_TRUE(settings::QMSettings::useThirdOrderDftb());

    _clearParser(parser);

    thirdOrderParseFunc({"third_order", "=", "no"}, 0);
    EXPECT_FALSE(settings::QMSettings::useThirdOrderDftb());
    EXPECT_TRUE(settings::QMSettings::isThirdOrderDftbSet());

    _clearParser(parser);

    ASSERT_THROW_MSG(
        thirdOrderParseFunc({"third_order", "=", "notABool"}, 0),
        exc::InputFileException,
        "Invalid value \"notABool\" for key \"third_order\" at line 0 in input "
        "file. Allowed values: on|off|true|false|yes|no"
    )
}

TEST_F(TestInputFileReader, parseHubbardDerivs)
{
    auto       parser  = input::QMInputParser();
    const auto funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("hubbard_derivs"));
    const auto& hubbardDerivsParseFunc = funcMap.at("hubbard_derivs");

    hubbardDerivsParseFunc({"hubbard_derivs", "=", "H:1.0,He:2.0"}, 0);

    const auto hubbardDerivs = settings::QMSettings::getHubbardDerivs();
    EXPECT_EQ(hubbardDerivs.size(), 2);
    EXPECT_EQ(hubbardDerivs.at("H"), 1.0);
    EXPECT_EQ(hubbardDerivs.at("He"), 2.0);

    _clearParser(parser);

    ASSERT_THROW_MSG(
        hubbardDerivsParseFunc({"hubbard_derivs", "=", "H:1.0,He"}, 0),
        exc::InputFileException,
        "Invalid value \"H:1.0,He\" for key \"hubbard_derivs\" at line 0 in "
        "input file. Value must be a comma-separated list of key:value pairs, "
        "where the key is a string and the value is a double."
    );

    _clearParser(parser);

    ASSERT_THROW_MSG(
        hubbardDerivsParseFunc({"hubbard_derivs", "=", "H:0.1junk"}, 0),
        exc::InputFileException,
        "Invalid value \"H:0.1junk\" for key \"hubbard_derivs\" at line 0 in "
        "input file. Value must be a comma-separated list of key:value pairs, "
        "where the key is a string and the value is a double."
    );

    _clearParser(parser);

    ASSERT_THROW_MSG(
        hubbardDerivsParseFunc({"hubbard_derivs", "=", "H:nan"}, 0),
        exc::InputFileException,
        "Invalid value \"H:nan\" for key \"hubbard_derivs\" at line 0 in input "
        "file. Value must be a comma-separated list of key:value pairs, where "
        "the key is a string and the value is a double."
    );
}

TEST_F(TestInputFileReader, parseXtbMethod)
{
    using enum QMMethod;

    auto       parser  = input::QMInputParser();
    const auto funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("xtb_method"));
    const auto& xtbMethodParseFunc = funcMap.at("xtb_method");

    xtbMethodParseFunc({"xtb_method", "=", "Gfn1-XTb"}, 0);
    EXPECT_EQ(settings::QMSettings::getXtbMethod(), XtbMethod::GFN1);

    _clearParser(parser);

    xtbMethodParseFunc({"xtb_method", "=", "gfN2-XTb"}, 0);
    EXPECT_EQ(settings::QMSettings::getXtbMethod(), XtbMethod::GFN2);

    _clearParser(parser);

    xtbMethodParseFunc({"xtb_method", "=", "iPEa1-XTb"}, 0);
    EXPECT_EQ(settings::QMSettings::getXtbMethod(), XtbMethod::IPEA1);

    _clearParser(parser);

    ASSERT_THROW_MSG(
        xtbMethodParseFunc({"xtb_method", "=", "notAnXtbMethod"}, 0),
        exc::InputFileException,
        "Invalid value \"notAnXtbMethod\" for key \"xtb_method\" at line 0 in "
        "input file. Allowed values: gfn1, gfn2, ipea1, gfn1-xtb, gfn2-xtb, "
        "ipea1-xtb, gfn1_xtb, gfn2_xtb, ipea1_xtb"
    )
}

TEST_F(TestInputFileReader, parseFennolModelPath)
{
    using enum QMMethod;

    auto       parser  = input::QMInputParser();
    const auto funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("fennol_model_path"));
    const auto& fennolModelPathParseFunc = funcMap.at("fennol_model_path");

    EXPECT_EQ(settings::QMSettings::getFennolModelPath(), "");
    fennolModelPathParseFunc(
        {"fennol_model_path", "=", "/pAth/to/fennol_model.fnx"},
        0
    );
    EXPECT_EQ(
        settings::QMSettings::getFennolModelPath(),
        "/pAth/to/fennol_model.fnx"
    );
}

TEST_F(TestInputFileReader, parseGPUPreprocessing)
{
    using enum QMMethod;

    auto       parser  = input::QMInputParser();
    const auto funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("gpu_preprocessing"));
    const auto& gpuPreprocessingParseFunc = funcMap.at("gpu_preprocessing");

    EXPECT_EQ(settings::QMSettings::useGPUPreprocessing(), true);
    gpuPreprocessingParseFunc({"GPU-Preprocessing", "=", "false"}, 0);
    EXPECT_EQ(settings::QMSettings::useGPUPreprocessing(), false);

    _clearParser(parser);

    gpuPreprocessingParseFunc({"gpu_preprocessing", "=", "on"}, 0);
    EXPECT_EQ(settings::QMSettings::useGPUPreprocessing(), true);

    _clearParser(parser);

    ASSERT_THROW_MSG(
        gpuPreprocessingParseFunc({"gpu_preprocessing", "=", "notABool"}, 0),
        exc::InputFileException,
        "Invalid value \"notABool\" for key \"gpu_preprocessing\" at line 0 in "
        "input file. Allowed values: on|off|true|false|yes|no"
    )
}

TEST_F(TestInputFileReader, processFennolKeywords)
{
    _inputFileReader->process({"fennol-model-path", "=", "model.fnx"});
    EXPECT_EQ(settings::QMSettings::getFennolModelPath(), "model.fnx");
    EXPECT_EQ(_inputFileReader->getKeywordCount("fennol_model_path"), 1);

    _inputFileReader->process({"GPU-Preprocessing", "=", "off"});
    EXPECT_EQ(settings::QMSettings::useGPUPreprocessing(), false);
    EXPECT_EQ(_inputFileReader->getKeywordCount("gpu_preprocessing"), 1);
}
