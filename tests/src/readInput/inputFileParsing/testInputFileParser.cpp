#include "exceptions.hpp"               // for InputFileException
#include "inputFileParser.hpp"          // for ParseFunc, checkCommand
#include "inputFileParserGeneral.hpp"   // for InputFileParserGeneral
#include "testInputFileReader.hpp"      // for TestInputFileReader
#include "throwWithMessage.hpp"         // for ASSERT_THROW_MSG

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <functional>      // for _Bind_front_t, bind_front
#include <gtest/gtest.h>   // for TestInfo (ptr only), EXPECT_EQ
#include <map>             // for map
#include <string>          // for string, allocator, basic_string
#include <vector>          // for vector

using namespace readInput;

/**
 * @brief tests checkCommand function
 *
 * @details if the number of arguments is not 3 it throws inputFileException
 *
 */
TEST_F(TestInputFileReader, checkCommand)
{
    auto lineElements = std::vector<std::string>{"test", "="};
    ASSERT_THROW_MSG(checkCommand(lineElements, 1),
                     customException::InputFileException,
                     "Invalid number of arguments at line 1 in input file");

    lineElements = std::vector<std::string>{"test", "=", "test2", "tooMany"};
    ASSERT_THROW_MSG(checkCommand(lineElements, 1),
                     customException::InputFileException,
                     "Invalid number of arguments at line 1 in input file");

    lineElements = std::vector<std::string>{"test", "=", "test2"};
    ASSERT_NO_THROW(checkCommand(lineElements, 1));
}

/**
 * @brief tests checkCommandArray function
 *
 * @details if the number of arguments is less than 3 it throws inputFileException
 *
 */
TEST_F(TestInputFileReader, checkCommandArray)
{
    auto lineElements = std::vector<std::string>{"test", "="};
    ASSERT_THROW_MSG(checkCommandArray(lineElements, 1),
                     customException::InputFileException,
                     "Invalid number of arguments at line 1 in input file");

    lineElements = std::vector<std::string>{"test", "=", "test2", "OK"};
    ASSERT_NO_THROW(checkCommandArray(lineElements, 1));

    lineElements = std::vector<std::string>{"test", "=", "test2"};
    ASSERT_NO_THROW(checkCommandArray(lineElements, 1));
}

/**
 * @brief tests checkEqualSign function
 *
 */
TEST_F(TestInputFileReader, equalSign)
{
    ASSERT_THROW_MSG(checkEqualSign("a", 1), customException::InputFileException, "Invalid command at line 1 in input file");

    ASSERT_NO_THROW(checkEqualSign("=", 1));
}

/**
 * @brief tests addKeyword function
 *
 * @details it adds a keyword to different keyword maps of an input file parser child object
 *
 */
TEST_F(TestInputFileReader, addKeyword)
{
    InputFileParserGeneral parser(_engine);
    const auto             initialSizeOfMaps = parser.getKeywordCountMap().size();

    parser.addKeyword("test", bind_front(&InputFileParserGeneral::parseJobType, parser), true);

    EXPECT_EQ(parser.getKeywordCountMap().size(), 1 + initialSizeOfMaps);
    EXPECT_EQ(parser.getKeywordCountMap().at("test"), 0);

    EXPECT_EQ(parser.getKeywordFuncMap().size(), 1 + initialSizeOfMaps);
    EXPECT_NO_THROW(parser.getKeywordFuncMap().at("test"));

    EXPECT_EQ(parser.getKeywordRequiredMap().size(), 1 + initialSizeOfMaps);
    EXPECT_EQ(parser.getKeywordRequiredMap().at("test"), true);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}