#include "exceptions.hpp"                // for InputFileException
#include "inputFileParser.hpp"           // for readInput
#include "inputFileParserManostat.hpp"   // for InputFileParserManostat
#include "manostatSettings.hpp"          // for ManostatSettings
#include "testInputFileReader.hpp"       // for TestInputFileReader
#include "throwWithMessage.hpp"          // for EXPECT_THROW_MSG

#include "gtest/gtest.h"   // for Message, TestPartResult, testing
#include <gtest/gtest.h>   // for TestInfo (ptr only), EXPECT_EQ
#include <string>          // for string, allocator, basic_string
#include <vector>          // for vector

using namespace readInput;

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
 * @details if the relaxation time of the manostat is negative it throws inputFileException
 *
 */
TEST_F(TestInputFileReader, ParseRelaxationTimeManostat)
{
    InputFileParserManostat  parser(*_engine);
    std::vector<std::string> lineElements = {"p_relaxation", "=", "0.1"};
    parser.parseManostatRelaxationTime(lineElements, 0);
    EXPECT_EQ(settings::ManostatSettings::getTauManostat(), 0.1);

    lineElements = {"p_relaxation", "=", "-100.0"};
    EXPECT_THROW_MSG(parser.parseManostatRelaxationTime(lineElements, 0),
                     customException::InputFileException,
                     "Relaxation time of manostat cannot be negative");
}

/**
 * @brief tests parsing the "manostat" command
 *
 * @details if the manostat is not valid it throws inputFileException - valid options are "none" and "berendsen"
 *
 */
TEST_F(TestInputFileReader, ParseManostat)
{
    InputFileParserManostat  parser(*_engine);
    std::vector<std::string> lineElements = {"manostat", "=", "none"};
    parser.parseManostat(lineElements, 0);
    EXPECT_EQ(settings::ManostatSettings::getManostatType(), settings::ManostatType::NONE);

    lineElements = {"manostat", "=", "berendsen"};
    parser.parseManostat(lineElements, 0);
    EXPECT_EQ(settings::ManostatSettings::getManostatType(), settings::ManostatType::BERENDSEN);

    lineElements = {"manostat", "=", "notValid"};
    EXPECT_THROW_MSG(parser.parseManostat(lineElements, 0),
                     customException::InputFileException,
                     "Invalid manostat \"notValid\" at line 0 in input file. Possible options are: berendsen and none");
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
        parser.parseCompressibility(lineElements, 0), customException::InputFileException, "Compressibility cannot be negative");
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}