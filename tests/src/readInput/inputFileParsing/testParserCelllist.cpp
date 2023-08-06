#include "exceptions.hpp"
#include "testInputFileReader.hpp"

using namespace std;
using namespace readInput;
using namespace ::testing;

TEST_F(TestInputFileReader, testParseCellListActivated)
{
    vector<string> lineElements = {"cell-list", "=", "off"};
    _inputFileReader->parseCellListActivated(lineElements);
    EXPECT_FALSE(_engine.getCellList().isActivated());
    lineElements = {"cell-list", "=", "on"};
    _inputFileReader->parseCellListActivated(lineElements);
    EXPECT_TRUE(_engine.getCellList().isActivated());
    lineElements = {"cell-list", "=", "1"};
    EXPECT_THROW(_inputFileReader->parseCellListActivated(lineElements), customException::InputFileException);
}

TEST_F(TestInputFileReader, testNumberOfCells)
{
    vector<string> lineElements = {"ncelss", "=", "3"};
    _inputFileReader->parseNumberOfCells(lineElements);
    EXPECT_EQ(_engine.getCellList().getNumberOfCells(), linearAlgebra::Vec3Dul(3, 3, 3));
    lineElements = {"ncelss", "=", "0"};
    EXPECT_THROW(_inputFileReader->parseNumberOfCells(lineElements), customException::InputFileException);
}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}