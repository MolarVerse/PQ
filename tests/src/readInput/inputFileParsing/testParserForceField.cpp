#include "exceptions.hpp"
#include "inputFileParserForceField.hpp"
#include "testInputFileReader.hpp"
#include "throwWithMessage.hpp"

using namespace std;
using namespace readInput;
using namespace ::testing;
using namespace customException;

/**
 * @brief tests parsing the "force-field" command
 *
 * @details possible options are on, off or bonded - otherwise throws inputFileException
 *
 */
TEST_F(TestInputFileReader, testParseForceField)
{
    InputFileParserForceField parser(_engine);
    vector<string>            lineElements = {"force-field", "=", "on"};
    parser.parseForceFieldType(lineElements, 0);
    EXPECT_TRUE(_engine.getForceFieldPtr()->isActivated());
    EXPECT_TRUE(_engine.getForceFieldPtr()->isNonCoulombicActivated());

    lineElements = {"force-field", "=", "off"};
    parser.parseForceFieldType(lineElements, 0);
    EXPECT_FALSE(_engine.getForceFieldPtr()->isActivated());
    EXPECT_FALSE(_engine.getForceFieldPtr()->isNonCoulombicActivated());

    lineElements = {"force-field", "=", "bonded"};
    parser.parseForceFieldType(lineElements, 0);
    EXPECT_TRUE(_engine.getForceFieldPtr()->isActivated());
    EXPECT_FALSE(_engine.getForceFieldPtr()->isNonCoulombicActivated());

    lineElements = {"forceField", "=", "notValid"};
    ASSERT_THROW_MSG(
        parser.parseForceFieldType(lineElements, 0),
        InputFileException,
        "Invalid force-field keyword \"notValid\" at line 0 in input file - possible keywords are \"on\", \"off\" or \"bonded\"");
}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}