#include "celllist.hpp"                  // for CellList
#include "engine.hpp"                    // for Engine
#include "exceptions.hpp"                // for InputFileException
#include "inputFileParser.hpp"           // for readInput
#include "inputFileParserCellList.hpp"   // for InputFileParserCellList
#include "testInputFileReader.hpp"       // for TestInputFileReader
#include "throwWithMessage.hpp"          // for EXPECT_THROW_MSG
#include "vector3d.hpp"                  // for Vec3Dul

#include "gtest/gtest.h"   // for Message, AssertionResult
#include <gtest/gtest.h>   // for TestInfo (ptr only), InitGoogleTest, RUN_ALL_TESTS, TEST_F, TestPartResult
#include <string>          // for string, allocator, basic_string
#include <vector>          // for vector

using namespace readInput;

/**
 * @brief tests parsing the "cell-list" command
 *
 * @details possible options are on or off - otherwise throws inputFileException
 *
 */
TEST_F(TestInputFileReader, parseCellListActivated)
{
    InputFileParserCellList  parser(*_engine);
    std::vector<std::string> lineElements = {"cell-list", "=", "off"};
    parser.parseCellListActivated(lineElements, 0);
    EXPECT_FALSE(_engine->getCellList().isActive());

    lineElements = {"cell-list", "=", "on"};
    parser.parseCellListActivated(lineElements, 0);
    EXPECT_TRUE(_engine->getCellList().isActive());

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
    InputFileParserCellList  parser(*_engine);
    std::vector<std::string> lineElements = {"cell-number", "=", "3"};
    parser.parseNumberOfCells(lineElements, 0);
    EXPECT_EQ(_engine->getCellList().getNumberOfCells(), linearAlgebra::Vec3Dul(3, 3, 3));

    lineElements = {"cell-number", "=", "0"};
    EXPECT_THROW_MSG(parser.parseNumberOfCells(lineElements, 0),
                     customException::InputFileException,
                     "Number of cells must be positive - number of cells = 0");
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}