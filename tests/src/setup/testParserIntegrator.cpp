#include "testInputFileReader.hpp"
#include "exceptions.hpp"

using namespace std;
using namespace Setup::InputFileReader;
using namespace ::testing;

TEST_F(TestInputFileReader, testParseIntegrator)
{
    vector<string> lineElements = {"integrator", "=", "v-verlet"};
    _inputFileReader->parseIntegrator(lineElements);
    EXPECT_EQ(_engine._integrator.getIntegratorType(), "VelocityVerlet");
}

TEST_F(TestInputFileReader, testIntegratorUnknown)
{
    vector<string> lineElements = {"integrator", "=", "unknown"};
    ASSERT_THROW(_inputFileReader->parseIntegrator(lineElements), InputFileException);
}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}