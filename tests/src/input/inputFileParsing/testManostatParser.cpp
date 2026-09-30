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

#include <gtest/gtest.h>   // for TestInfo (ptr only), EXPECT_EQ

#include <string>   // for string, allocator, basic_string
#include <vector>   // for vector

#include "exceptions.hpp"   // for InputFileException
#include "manostatInputParser.hpp"
#include "manostatSettings.hpp"      // for ManostatSettings
#include "testInputFileReader.hpp"   // for TestInputFileReader
#include "throwWithMessage.hpp"      // for EXPECT_THROW_MSG

/**
 * @brief tests parsing the "pressure" command
 *
 */
TEST_F(TestInputFileReader, ParsePressure)
{
    input::ManostatInputParser parser;
    const auto                 funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("pressure"));
    const auto& parseFunc = funcMap.at("pressure");

    std::vector<std::string> lineElements = {"pressure", "=", "300.0"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::ManostatSettings::getTargetPressure(), 300.0);

    _clearParser(parser);

    lineElements = {"pressure", "=", "nan"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"nan\" for key \"pressure\" at line 0 in input file: "
        "failed validation with message Value must not be NaN"
    );
}

/**
 * @brief tests parsing the "p_relaxation" command
 *
 * @details if the relaxation time of the manostat is negative it throws
 * inputFileException
 *
 */
TEST_F(TestInputFileReader, ParseRelaxationTimeManostat)
{
    input::ManostatInputParser parser;
    const auto                 funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("p_relaxation"));
    const auto& parseFunc = funcMap.at("p_relaxation");

    std::vector<std::string> lineElements = {"p_relaxation", "=", "0.1"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::ManostatSettings::getTauManostat(), 0.1);

    _clearParser(parser);

    lineElements = {"p_relaxation", "=", "-100.0"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"-100.0\" for key \"p_relaxation\" at line 0 in input "
        "file: "
        "failed validation with message Value must be between 0 and "
        "1.7976931348623156e+305"
    );

    _clearParser(parser);

    lineElements = {"p_relaxation", "=", "0"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"0\" for key \"p_relaxation\" at line 0 in input file: "
        "failed validation with message Value must be between 0 and "
        "1.7976931348623156e+305"
    );

    _clearParser(parser);

    lineElements = {"p_relaxation", "=", "1e308"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"1e308\" for key \"p_relaxation\" at line 0 in input "
        "file: failed validation with message Value must be between 0 and "
        "1.7976931348623156e+305"
    );
}

/**
 * @brief tests parsing the "manostat" command
 *
 * @details if the manostat is not valid it throws inputFileException - valid
 * options are "none" and "berendsen"
 *
 */
TEST_F(TestInputFileReader, ParseManostat)
{
    input::ManostatInputParser parser;
    const auto                 funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("manostat"));
    const auto& parseFunc = funcMap.at("manostat");

    std::vector<std::string> lineElements = {"manostat", "=", "none"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(
        settings::ManostatSettings::getManostatType(),
        ManostatType::NONE
    );

    _clearParser(parser);

    lineElements = {"manostat", "=", "berendsen"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(
        settings::ManostatSettings::getManostatType(),
        ManostatType::BERENDSEN
    );

    _clearParser(parser);

    lineElements = {"manostat", "=", "stochastic_rescaling"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(
        settings::ManostatSettings::getManostatType(),
        ManostatType::STOCHASTIC_RESCALING
    );

    _clearParser(parser);

    lineElements = {"manostat", "=", "notValid"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"notValid\" for key \"manostat\" at line 0 in input "
        "file. Allowed values: none, berendsen, stochastic_rescaling"
    );
}

/**
 * @brief tests parsing the "compressibility" command
 *
 * @details if the compressibility is negative it throws inputFileException
 *
 */
TEST_F(TestInputFileReader, ParseCompressibility)
{
    input::ManostatInputParser parser;
    const auto                 funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("compressibility"));
    const auto& parseFunc = funcMap.at("compressibility");

    std::vector<std::string> lineElements = {"compressibility", "=", "0.1"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::ManostatSettings::getCompressibility(), 0.1);

    _clearParser(parser);

    lineElements = {"compressibility", "=", "-0.1"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"-0.1\" for key \"compressibility\" at line 0 in input "
        "file: failed validation with message Value must be greater than 0"
    );

    _clearParser(parser);

    lineElements = {"compressibility", "=", "inf"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"inf\" for key \"compressibility\" at line 0 in input "
        "file: failed validation with message Value must not be infinite"
    );
}

/**
 * @brief tests parsing the "isotropy" command
 *
 */
TEST_F(TestInputFileReader, ParseIsotropy)
{
    input::ManostatInputParser parser;
    const auto                 funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("isotropy"));
    const auto& parseFunc = funcMap.at("isotropy");

    std::vector<std::string> lineElements = {"isotropy", "=", "isotropic"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::ManostatSettings::getIsotropy(), Isotropy::ISOTROPIC);

    _clearParser(parser);

    lineElements = {"isotropy", "=", "anisotropic"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::ManostatSettings::getIsotropy(), Isotropy::ANISOTROPIC);

    _clearParser(parser);

    lineElements = {"isotropy", "=", "full_anisotropic"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(
        settings::ManostatSettings::getIsotropy(),
        Isotropy::FULL_ANISOTROPIC
    );

    _clearParser(parser);

    lineElements = {"isotropy", "=", "xz"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(
        settings::ManostatSettings::getIsotropy(),
        Isotropy::SEMI_ISOTROPIC_XZ
    );

    _clearParser(parser);

    lineElements = {"isotropy", "=", "zx"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(
        settings::ManostatSettings::getIsotropy(),
        Isotropy::SEMI_ISOTROPIC_XZ
    );

    _clearParser(parser);

    lineElements = {"isotropy", "=", "yz"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(
        settings::ManostatSettings::getIsotropy(),
        Isotropy::SEMI_ISOTROPIC_YZ
    );

    _clearParser(parser);

    lineElements = {"isotropy", "=", "zy"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(
        settings::ManostatSettings::getIsotropy(),
        Isotropy::SEMI_ISOTROPIC_YZ
    );

    _clearParser(parser);

    lineElements = {"isotropy", "=", "xy"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(
        settings::ManostatSettings::getIsotropy(),
        Isotropy::SEMI_ISOTROPIC_XY
    );

    _clearParser(parser);

    lineElements = {"isotropy", "=", "yx"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(
        settings::ManostatSettings::getIsotropy(),
        Isotropy::SEMI_ISOTROPIC_XY
    );

    _clearParser(parser);

    lineElements = {"isotropy", "=", "notValid"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"notValid\" for key \"isotropy\" at line 0 in input "
        "file. Allowed values: isotropic, semi_isotropic_xy, "
        "semi_isotropic_xz, semi_isotropic_yz, anisotropic, full_anisotropic, "
        "xy, yx, xz, zx, yz, zy"
    );
}

/**
 * @brief tests parsing the "fixed_axis" command
 *
 */
TEST_F(TestInputFileReader, ParseFixedAxis)
{
    input::ManostatInputParser parser;
    const auto                 funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("fixed_axis"));
    const auto& parseFunc = funcMap.at("fixed_axis");

    std::vector<std::string> lineElements = {"fixed_axis", "=", "none"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::ManostatSettings::getFixedAxis(), FixedAxis::NONE);

    _clearParser(parser);

    lineElements = {"fixed_axis", "=", "x"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::ManostatSettings::getFixedAxis(), FixedAxis::X);

    _clearParser(parser);

    lineElements = {"fixed_axis", "=", "y"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::ManostatSettings::getFixedAxis(), FixedAxis::Y);

    _clearParser(parser);

    lineElements = {"fixed_axis", "=", "z"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::ManostatSettings::getFixedAxis(), FixedAxis::Z);

    _clearParser(parser);

    lineElements = {"fixed_axis", "=", "xy"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::ManostatSettings::getFixedAxis(), FixedAxis::XY);
    EXPECT_EQ(settings::ManostatSettings::getFixedAxis(), FixedAxis::XY);

    _clearParser(parser);

    lineElements = {"fixed_axis", "=", "yx"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::ManostatSettings::getFixedAxis(), FixedAxis::XY);

    _clearParser(parser);

    lineElements = {"fixed_axis", "=", "xz"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::ManostatSettings::getFixedAxis(), FixedAxis::XZ);

    _clearParser(parser);

    lineElements = {"fixed_axis", "=", "zx"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::ManostatSettings::getFixedAxis(), FixedAxis::XZ);

    _clearParser(parser);

    lineElements = {"fixed_axis", "=", "yz"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::ManostatSettings::getFixedAxis(), FixedAxis::YZ);

    _clearParser(parser);

    lineElements = {"fixed_axis", "=", "zy"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::ManostatSettings::getFixedAxis(), FixedAxis::YZ);

    _clearParser(parser);

    lineElements = {"fixed_axis", "=", "all"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::ManostatSettings::getFixedAxis(), FixedAxis::ALL);

    _clearParser(parser);

    lineElements = {"fixed_axis", "=", "xyz"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::ManostatSettings::getFixedAxis(), FixedAxis::ALL);

    _clearParser(parser);

    lineElements = {"fixed_axis", "=", "notValid"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"notValid\" for key \"fixed_axis\" at line 0 in input "
        "file. Allowed values: none, x, y, z, xy, xz, yz, all, yx, zx, zy, xyz"
    );
}
