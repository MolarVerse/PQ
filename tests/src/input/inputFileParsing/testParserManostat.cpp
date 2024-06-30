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

#include "exceptions.hpp"                // for InputFileException
#include "gtest/gtest.h"                 // for Message, TestPartResult, testing
#include "inputFileParser.hpp"           // for readInput
#include "inputFileParserManostat.hpp"   // for InputFileParserManostat
#include "manostatSettings.hpp"          // for ManostatSettings
#include "testInputFileReader.hpp"       // for TestInputFileReader
#include "throwWithMessage.hpp"          // for EXPECT_THROW_MSG

using namespace input;

/**
 * @brief tests parsing the "pressure" command
 *
 */
TEST_F(TestInputFileReader, ParsePressure)
{
    EXPECT_EQ(settings::ManostatSettings::isPressureSet(), false);

    InputFileParserManostat  parser(*_engine);
    std::vector<std::string> lineElements = {"pressure", "=", "300.0"};
    parser.parsePressure(lineElements, 0);

    EXPECT_EQ(settings::ManostatSettings::getTargetPressure(), 300.0);
    EXPECT_EQ(settings::ManostatSettings::isPressureSet(), true);
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
    InputFileParserManostat  parser(*_engine);
    std::vector<std::string> lineElements = {"p_relaxation", "=", "0.1"};
    parser.parseManostatRelaxationTime(lineElements, 0);
    EXPECT_EQ(settings::ManostatSettings::getTauManostat(), 0.1);

    lineElements = {"p_relaxation", "=", "-100.0"};
    EXPECT_THROW_MSG(
        parser.parseManostatRelaxationTime(lineElements, 0),
        customException::InputFileException,
        "Relaxation time of manostat cannot be negative"
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
    InputFileParserManostat  parser(*_engine);
    std::vector<std::string> lineElements = {"manostat", "=", "none"};
    parser.parseManostat(lineElements, 0);
    EXPECT_EQ(
        settings::ManostatSettings::getManostatType(),
        settings::ManostatType::NONE
    );

    lineElements = {"manostat", "=", "berendsen"};
    parser.parseManostat(lineElements, 0);
    EXPECT_EQ(
        settings::ManostatSettings::getManostatType(),
        settings::ManostatType::BERENDSEN
    );

    lineElements = {"manostat", "=", "stochastic_rescaling"};
    parser.parseManostat(lineElements, 0);
    EXPECT_EQ(
        settings::ManostatSettings::getManostatType(),
        settings::ManostatType::STOCHASTIC_RESCALING
    );

    lineElements = {"manostat", "=", "notValid"};
    EXPECT_THROW_MSG(
        parser.parseManostat(lineElements, 0),
        customException::InputFileException,
        "Invalid manostat \"notValid\" at line 0 in input file.\n"
        "Possible options are: berendsen, stochastic_rescaling and none"
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
    InputFileParserManostat  parser(*_engine);
    std::vector<std::string> lineElements = {"compressibility", "=", "0.1"};
    parser.parseCompressibility(lineElements, 0);
    EXPECT_EQ(settings::ManostatSettings::getCompressibility(), 0.1);

    lineElements = {"compressibility", "=", "-0.1"};
    EXPECT_THROW_MSG(
        parser.parseCompressibility(lineElements, 0),
        customException::InputFileException,
        "Compressibility cannot be negative"
    );
}

/**
 * @brief tests parsing the "isotropy" command
 *
 */
TEST_F(TestInputFileReader, ParseIsotropy)
{
    InputFileParserManostat  parser(*_engine);
    std::vector<std::string> lineElements = {"isotropy", "=", "isotropic"};
    parser.parseIsotropy(lineElements, 0);
    EXPECT_EQ(
        settings::ManostatSettings::getIsotropy(),
        settings::Isotropy::ISOTROPIC
    );

    lineElements = {"isotropy", "=", "anisotropic"};
    parser.parseIsotropy(lineElements, 0);
    EXPECT_EQ(
        settings::ManostatSettings::getIsotropy(),
        settings::Isotropy::ANISOTROPIC
    );

    lineElements = {"isotropy", "=", "full_anisotropic"};
    parser.parseIsotropy(lineElements, 0);
    EXPECT_EQ(
        settings::ManostatSettings::getIsotropy(),
        settings::Isotropy::FULL_ANISOTROPIC
    );

    lineElements = {"isotropy", "=", "xz"};
    parser.parseIsotropy(lineElements, 0);
    EXPECT_EQ(
        settings::ManostatSettings::getIsotropy(),
        settings::Isotropy::SEMI_ISOTROPIC
    );
    EXPECT_EQ(settings::ManostatSettings::get2DIsotropicAxes()[0], 0);
    EXPECT_EQ(settings::ManostatSettings::get2DIsotropicAxes()[1], 2);
    EXPECT_EQ(settings::ManostatSettings::get2DAnisotropicAxis(), 1);

    lineElements = {"isotropy", "=", "zx"};
    parser.parseIsotropy(lineElements, 0);
    EXPECT_EQ(
        settings::ManostatSettings::getIsotropy(),
        settings::Isotropy::SEMI_ISOTROPIC
    );
    EXPECT_EQ(settings::ManostatSettings::get2DIsotropicAxes()[0], 0);
    EXPECT_EQ(settings::ManostatSettings::get2DIsotropicAxes()[1], 2);
    EXPECT_EQ(settings::ManostatSettings::get2DAnisotropicAxis(), 1);

    lineElements = {"isotropy", "=", "yz"};
    parser.parseIsotropy(lineElements, 0);
    EXPECT_EQ(
        settings::ManostatSettings::getIsotropy(),
        settings::Isotropy::SEMI_ISOTROPIC
    );
    EXPECT_EQ(settings::ManostatSettings::get2DIsotropicAxes()[0], 1);
    EXPECT_EQ(settings::ManostatSettings::get2DIsotropicAxes()[1], 2);
    EXPECT_EQ(settings::ManostatSettings::get2DAnisotropicAxis(), 0);

    lineElements = {"isotropy", "=", "zy"};
    parser.parseIsotropy(lineElements, 0);
    EXPECT_EQ(
        settings::ManostatSettings::getIsotropy(),
        settings::Isotropy::SEMI_ISOTROPIC
    );
    EXPECT_EQ(settings::ManostatSettings::get2DIsotropicAxes()[0], 1);
    EXPECT_EQ(settings::ManostatSettings::get2DIsotropicAxes()[1], 2);
    EXPECT_EQ(settings::ManostatSettings::get2DAnisotropicAxis(), 0);

    lineElements = {"isotropy", "=", "xy"};
    parser.parseIsotropy(lineElements, 0);
    EXPECT_EQ(
        settings::ManostatSettings::getIsotropy(),
        settings::Isotropy::SEMI_ISOTROPIC
    );
    EXPECT_EQ(settings::ManostatSettings::get2DIsotropicAxes()[0], 0);
    EXPECT_EQ(settings::ManostatSettings::get2DIsotropicAxes()[1], 1);
    EXPECT_EQ(settings::ManostatSettings::get2DAnisotropicAxis(), 2);

    lineElements = {"isotropy", "=", "yx"};
    parser.parseIsotropy(lineElements, 0);
    EXPECT_EQ(
        settings::ManostatSettings::getIsotropy(),
        settings::Isotropy::SEMI_ISOTROPIC
    );
    EXPECT_EQ(settings::ManostatSettings::get2DIsotropicAxes()[0], 0);
    EXPECT_EQ(settings::ManostatSettings::get2DIsotropicAxes()[1], 1);
    EXPECT_EQ(settings::ManostatSettings::get2DAnisotropicAxis(), 2);

    lineElements = {"isotropy", "=", "notValid"};
    EXPECT_THROW_MSG(
        parser.parseIsotropy(lineElements, 0),
        customException::InputFileException,
        "Invalid isotropy \"notValid\" at line 0 in input file.\n"
        "Possible options are: isotropic, xy, xz, yz, anisotropic and "
        "full_anisotropic"
    );
}
