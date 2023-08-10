#include "exceptions.hpp"
#include "inputFileParserManostat.hpp"
#include "testInputFileReader.hpp"
#include "throwWithMessage.hpp"

using namespace std;
using namespace readInput;
using namespace ::testing;

/**
 * @brief tests parsing the "pressure" command
 *
 */
TEST_F(TestInputFileReader, ParsePressure)
{
    InputFileParserManostat parser(_engine);
    vector<string>          lineElements = {"pressure", "=", "300.0"};
    parser.parsePressure(lineElements, 0);
    EXPECT_EQ(_engine.getSettings().getPressure(), 300.0);
}

/**
 * @brief tests parsing the "p_relaxation" command
 *
 * @details if the relaxation time of the manostat is negative it throws inputFileException
 *
 */
TEST_F(TestInputFileReader, ParseRelaxationTimeManostat)
{
    InputFileParserManostat parser(_engine);
    vector<string>          lineElements = {"p_relaxation", "=", "0.1"};
    parser.parseManostatRelaxationTime(lineElements, 0);
    EXPECT_EQ(_engine.getSettings().getTauManostat(), 0.1);

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
    InputFileParserManostat parser(_engine);
    vector<string>          lineElements = {"manostat", "=", "none"};
    parser.parseManostat(lineElements, 0);
    EXPECT_EQ(_engine.getSettings().getManostat(), "none");

    lineElements = {"manostat", "=", "berendsen"};
    parser.parseManostat(lineElements, 0);
    EXPECT_EQ(_engine.getSettings().getManostat(), "berendsen");

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
    InputFileParserManostat parser(_engine);
    vector<string>          lineElements = {"compressibility", "=", "0.0"};
    parser.parseCompressibility(lineElements, 0);
    EXPECT_EQ(_engine.getSettings().getCompressibility(), 0.0);

    lineElements = {"compressibility", "=", "0.1"};
    parser.parseCompressibility(lineElements, 0);
    EXPECT_EQ(_engine.getSettings().getCompressibility(), 0.1);

    lineElements = {"compressibility", "=", "-0.1"};
    EXPECT_THROW_MSG(
        parser.parseCompressibility(lineElements, 0), customException::InputFileException, "Compressibility cannot be negative");
}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}