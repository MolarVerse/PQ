#include "engine.hpp"                      // for Engine
#include "exceptions.hpp"                  // for InputFileException, customException
#include "inputFileParser.hpp"             // for readInput
#include "inputFileParserIntegrator.hpp"   // for InputFileParserIntegrator
#include "integrator.hpp"                  // for Integrator
#include "testInputFileReader.hpp"         // for TestInputFileReader
#include "throwWithMessage.hpp"            // for ASSERT_THROW_MSG

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <gtest/gtest.h>   // for InitGoogleTest, RUN_ALL_TESTS
#include <string>          // for string, allocator, basic_string
#include <vector>          // for vector

using namespace readInput;

/**
 * @brief tests parsing the "integrator" command
 *
 * @details possible options are v-verlet - otherwise throws inputFileException
 *
 */
TEST_F(TestInputFileReader, testParseIntegrator)
{
    InputFileParserIntegrator parser(_engine);
    std::vector<std::string>  lineElements = {"integrator", "=", "v-verlet"};
    parser.parseIntegrator(lineElements, 0);
    EXPECT_EQ(_engine.getIntegrator().getIntegratorType(), "VelocityVerlet");

    lineElements = {"integrator", "=", "notValid"};
    ASSERT_THROW_MSG(parser.parseIntegrator(lineElements, 0),
                     customException::InputFileException,
                     "Invalid integrator \"notValid\" at line 0 in input file");
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}