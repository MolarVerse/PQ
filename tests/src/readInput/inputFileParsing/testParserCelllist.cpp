#include "exceptions.hpp"
#include "testInputFileReader.hpp"
#include "throwWithMessage.hpp"

using namespace std;
using namespace readInput;
using namespace ::testing;

/**
 * @brief tests parsing the "cell-list" command
 *
 * @details possible options are on or off - otherwise throws inputFileException
 *
 */
TEST_F(TestInputFileReader, parseCellListActivated)
{
    InputFileParserCellList parser(_engine);
    vector<string>          lineElements = {"cell-list", "=", "off"};
    parser.parseCellListActivated(lineElements, 0);
    EXPECT_FALSE(_engine.getCellList().isActivated());

    lineElements = {"cell-list", "=", "on"};
    parser.parseCellListActivated(lineElements, 0);
    EXPECT_TRUE(_engine.getCellList().isActivated());

    lineElements = {"cell-list", "=", "notValid"};
    EXPECT_THROW_MSG(parser.parseCellListActivated(lineElements, 0),
                     customException::InputFileException,
                     R"(Invalid cell-list keyword "notValid" at line 0 in input file\n Possible keywords are "on" and "off")");
}

/**
 * @brief tests parsing the "cell-number" command
 *
 * @details if the number of cells is negative or 0, throws inputFileException
 *
 */
TEST_F(TestInputFileReader, numberOfCells)
{
    InputFileParserCellList parser(_engine);
    vector<string>          lineElements = {"cell-number", "=", "3"};
    parser.parseNumberOfCells(lineElements, 0);
    EXPECT_EQ(_engine.getCellList().getNumberOfCells(), linearAlgebra::Vec3Dul(3, 3, 3));

    lineElements = {"cell-number", "=", "0"};
    EXPECT_THROW_MSG(parser.parseNumberOfCells(lineElements, 0),
                     customException::InputFileException,
                     "Number of cells must be positive - number of cells = 0");
}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}