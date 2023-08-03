#include "exceptions.hpp"
#include "testInputFileReader.hpp"

using namespace std;
using namespace readInput;
using namespace ::testing;

/**
 * @brief tests the parsing of the noncoulomb type
 *
 * @details possible options are "none", "lj" and "buck"
 *
 */
TEST_F(TestInputFileReader, testParseNonCoulombType)
{
    vector<string> lineElements = {"noncoulomb", "=", "none"};
    _inputFileReader->parseNonCoulombType(lineElements);
    EXPECT_EQ(_engine.getSettings().getNonCoulombType(), "none");

    lineElements = {"noncoulomb", "=", "lj"};
    _inputFileReader->parseNonCoulombType(lineElements);
    EXPECT_EQ(_engine.getSettings().getNonCoulombType(), "lj");

    lineElements = {"noncoulomb", "=", "buck"};
    _inputFileReader->parseNonCoulombType(lineElements);
    EXPECT_EQ(_engine.getSettings().getNonCoulombType(), "buck");

    lineElements = {"coulomb", "=", "notValid"};
    EXPECT_THROW(_inputFileReader->parseNonCoulombType(lineElements), customException::InputFileException);
}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}