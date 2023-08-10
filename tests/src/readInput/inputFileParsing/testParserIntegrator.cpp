#include "exceptions.hpp"
#include "inputFileParserIntegrator.hpp"
#include "testInputFileReader.hpp"
#include "throwWithMessage.hpp"

using namespace std;
using namespace readInput;
using namespace ::testing;
using namespace customException;

/**
 * @brief tests parsing the "integrator" command
 *
 * @details possible options are v-verlet - otherwise throws inputFileException
 *
 */
TEST_F(TestInputFileReader, testParseIntegrator)
{
    InputFileParserIntegrator parser(_engine);
    vector<string>            lineElements = {"integrator", "=", "v-verlet"};
    parser.parseIntegrator(lineElements, 0);
    EXPECT_EQ(_engine.getIntegrator().getIntegratorType(), "VelocityVerlet");

    lineElements = {"integrator", "=", "notValid"};
    ASSERT_THROW_MSG(
        parser.parseIntegrator(lineElements, 0), InputFileException, "Invalid integrator \"notValid\" at line 0 in input file");
}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}