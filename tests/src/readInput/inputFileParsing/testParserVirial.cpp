#include "exceptions.hpp"
#include "inputFileParserVirial.hpp"
#include "testInputFileReader.hpp"
#include "throwWithMessage.hpp"

using namespace std;
using namespace readInput;
using namespace ::testing;

/**
 * @brief tests parsing the "virial" command
 *
 * @details possible options are atomic or molecular - otherwise throws inputFileException
 *
 */
TEST_F(TestInputFileReader, testParseVirial)
{
    InputFileParserVirial parser(_engine);
    vector<string>        lineElements = {"virial", "=", "atomic"};
    parser.parseVirial(lineElements, 0);
    EXPECT_EQ(_engine.getVirial().getVirialType(), "atomic");

    lineElements = {"virial", "=", "molecular"};
    parser.parseVirial(lineElements, 0);
    EXPECT_EQ(_engine.getVirial().getVirialType(), "molecular");

    lineElements = {"virial", "=", "notValid"};
    EXPECT_THROW_MSG(parser.parseVirial(lineElements, 0),
                     customException::InputFileException,
                     "Invalid virial setting \"notValid\" at line 0 in input file");
}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}