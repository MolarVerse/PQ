#include "engine.hpp"                      // for Engine
#include "exceptions.hpp"                  // for InputFileException, customException
#include "forceField.hpp"                  // for ForceField
#include "inputFileParser.hpp"             // for readInput
#include "inputFileParserForceField.hpp"   // for InputFileParserForceField
#include "testInputFileReader.hpp"         // for TestInputFileReader
#include "throwWithMessage.hpp"            // for ASSERT_THROW_MSG

#include "gtest/gtest.h"   // for AssertionResult, Message
#include <gtest/gtest.h>   // for EXPECT_FALSE, EXPECT_TRUE
#include <string>          // for string, allocator, basic_string
#include <vector>          // for vector

using namespace readInput;

/**
 * @brief tests parsing the "force-field" command
 *
 * @details possible options are on, off or bonded - otherwise throws inputFileException
 *
 */
TEST_F(TestInputFileReader, testParseForceField)
{
    InputFileParserForceField parser(_engine);
    std::vector<std::string>  lineElements = {"force-field", "=", "on"};
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
        customException::InputFileException,
        "Invalid force-field keyword \"notValid\" at line 0 in input file - possible keywords are \"on\", \"off\" or \"bonded\"");
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}