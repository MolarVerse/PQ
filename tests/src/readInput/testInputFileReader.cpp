#include "testInputFileReader.hpp"

#include "exceptions.hpp"        // for InputFileException, customException
#include "inputFileParser.hpp"   // for readInput

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <gtest/gtest.h>   // for Message, TestPartResult
#include <map>             // for map
#include <sstream>         // for basic_istringstream
#include <vector>          // for vector, _Bit_iterator, _Bit_reference

using namespace readInput;

static void readKeywordList(const std::string &filename, std::vector<std::string> &keywords, std::vector<bool> &required)
{
    std::string   line;
    std::ifstream inputFile(filename);

    while (getline(inputFile, line))
    {

        std::string keyword;
        std::string requiredString = "";
        bool        requiredBool   = false;

        if (std::istringstream(line).str().empty())
            continue;
        std::istringstream(line) >> keyword >> requiredString;
        std::istringstream(requiredString) >> std::boolalpha >> requiredBool;

        keywords.push_back(keyword);
        required.push_back(requiredBool);
    }
}

TEST_F(TestInputFileReader, testAddKeyword)
{
    std::vector<std::string> keywordsRef(0);
    std::vector<bool>        requiredRef(0);

    ::readKeywordList("data/inputFileReader/keywordList.txt", keywordsRef, requiredRef);

    EXPECT_EQ(_inputFileReader->getKeywordCountMap().size(), keywordsRef.size());
    EXPECT_EQ(_inputFileReader->getKeywordRequiredMap().size(), keywordsRef.size());
    EXPECT_EQ(_inputFileReader->getKeywordFuncMap().size(), keywordsRef.size());

    for (size_t i = 0; i < keywordsRef.size(); ++i)
    {
        std::string keyword  = keywordsRef[i];
        bool        required = requiredRef[i];

        EXPECT_EQ(_inputFileReader->getKeywordCount(keyword), 0);
        EXPECT_EQ(_inputFileReader->getKeywordRequired(keyword), required);
    }
}

TEST_F(TestInputFileReader, testNotAValidKeyword)
{
    auto lineElements = std::vector<std::string>{"notAValidKeyword", "=", "1"};
    ASSERT_THROW(_inputFileReader->process(lineElements), customException::InputFileException);
}

TEST_F(TestInputFileReader, testProcess)
{
    auto lineElements = std::vector<std::string>{"nstep", "=", "1000"};
    _inputFileReader->process(lineElements);
    EXPECT_EQ(_inputFileReader->getKeywordCount(lineElements[0]), 1);
}

TEST_F(TestInputFileReader, testRead)
{
    std::string filename = "data/inputFileReader/inputFile.txt";
    _inputFileReader->setFilename(filename);
    ASSERT_NO_THROW(_inputFileReader->read());
}

TEST_F(TestInputFileReader, testReadFileNotFound)
{
    std::string filename = "data/inputFileReader/inputFileNotFound.txt";
    _inputFileReader->setFilename(filename);
    ASSERT_THROW(_inputFileReader->read(), customException::InputFileException);
}

TEST_F(TestInputFileReader, testReadInputFileFunction)
{
    std::string filename = "data/inputFileReader/inputFile.txt";
    ASSERT_NO_THROW(readInputFile(filename, _engine));
}

TEST_F(TestInputFileReader, testPostProcessRequiredFail)
{
    std::vector<std::string> keywordsRef(0);
    std::vector<bool>        requiredRef(0);

    ::readKeywordList("data/inputFileReader/keywordList.txt", keywordsRef, requiredRef);

    std::vector<size_t> requiredIndex(0);

    for (size_t i = 0; i < keywordsRef.size(); ++i)
    {
        std::string keyword  = keywordsRef[i];
        bool        required = requiredRef[i];

        if (required)
        {
            requiredIndex.push_back(i);
            _inputFileReader->setKeywordCount(keyword, 1);
        }
    }

    for (auto const &index : requiredIndex)
    {
        const auto &keyword = keywordsRef[index];
        _inputFileReader->setKeywordCount(keyword, 0);
        ASSERT_THROW(_inputFileReader->postProcess(), customException::InputFileException);
        _inputFileReader->setKeywordCount(keyword, 1);
    }
}

TEST_F(TestInputFileReader, testPostProcessCountToOftenFail)
{
    std::vector<std::string> keywordsRef(0);
    std::vector<bool>        requiredRef(0);

    ::readKeywordList("data/inputFileReader/keywordList.txt", keywordsRef, requiredRef);

    std::vector<size_t> requiredIndex(0);

    for (size_t i = 0; i < keywordsRef.size(); ++i)
    {
        const auto &keyword  = keywordsRef[i];
        bool        required = requiredRef[i];

        if (required)
        {
            requiredIndex.push_back(i);
            _inputFileReader->setKeywordCount(keyword, 1);
        }
    }

    for (const auto &index : requiredIndex)
    {
        if (index != 1)
        {
            const auto &keyword = keywordsRef[index];
            _inputFileReader->setKeywordCount(keyword, index);
            ASSERT_THROW(_inputFileReader->postProcess(), customException::InputFileException);
            _inputFileReader->setKeywordCount(keyword, 1);
        }
    }
}

TEST_F(TestInputFileReader, testMoldescriptorFileProcess)
{
    std::vector<std::string> keywordsRef(0);
    std::vector<bool>        requiredRef(0);

    ::readKeywordList("data/inputFileReader/keywordList.txt", keywordsRef, requiredRef);

    std::vector<size_t> requiredIndex(0);

    for (size_t i = 0; i < keywordsRef.size(); ++i)
    {
        const auto &keyword  = keywordsRef[i];
        bool        required = requiredRef[i];

        if (required)
        {
            requiredIndex.push_back(i);
            _inputFileReader->setKeywordCount(keyword, 1);
        }
    }

    EXPECT_NO_THROW(_inputFileReader->postProcess());
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}