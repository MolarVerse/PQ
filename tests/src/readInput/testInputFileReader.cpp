#include "testInputFileReader.hpp"

#include "exceptions.hpp"        // for InputFileException, customException
#include "inputFileParser.hpp"   // for readInput
#include "settings.hpp"          // for Settings
#include "stringUtilities.hpp"   // for getLineCommands, utilities

#include "gmock/gmock.h"   // for ElementsAre
#include "gtest/gtest.h"   // for Message, TestPartResult
#include <map>             // for map
#include <sstream>         // for basic_istringstream
#include <vector>          // for vector, _Bit_iterator, _Bit_reference

using namespace std;
using namespace readInput;
using namespace utilities;
using namespace ::testing;
using namespace customException;

void readKeywordList(const string &filename, vector<string> &keywords, vector<bool> &required)
{
    string   line;
    ifstream inputFile(filename);

    while (getline(inputFile, line))
    {

        string keyword;
        string requiredString = "";
        bool   requiredBool   = false;

        if (istringstream(line).str().empty())
            continue;
        istringstream(line) >> keyword >> requiredString;
        istringstream(requiredString) >> std::boolalpha >> requiredBool;

        keywords.push_back(keyword);
        required.push_back(requiredBool);
    }
}

TEST_F(TestInputFileReader, testAddKeyword)
{
    vector<string> keywordsRef(0);
    vector<bool>   requiredRef(0);

    ::readKeywordList("data/inputFileReader/keywordList.txt", keywordsRef, requiredRef);

    EXPECT_EQ(_inputFileReader->getKeywordCountMap().size(), keywordsRef.size());
    EXPECT_EQ(_inputFileReader->getKeywordRequiredMap().size(), keywordsRef.size());
    EXPECT_EQ(_inputFileReader->getKeywordFuncMap().size(), keywordsRef.size());

    for (size_t i = 0; i < keywordsRef.size(); ++i)
    {
        string keyword  = keywordsRef[i];
        bool   required = requiredRef[i];

        EXPECT_EQ(_inputFileReader->getKeywordCount(keyword), 0);
        EXPECT_EQ(_inputFileReader->getKeywordRequired(keyword), required);
    }
}

TEST_F(TestInputFileReader, testGetLineCommands)
{
    auto *line = "nstep = 1";
    ASSERT_THROW(getLineCommands(line, 1), InputFileException);

    line = "nstep = 1;";
    ASSERT_THAT(getLineCommands(line, 1), ElementsAre("nstep = 1", ""));

    line = "nstep = 1; nstep = 2";
    ASSERT_THROW(getLineCommands(line, 1), InputFileException);

    line = "nstep = 1; nstep = 2;";
    ASSERT_THAT(getLineCommands(line, 1), ElementsAre("nstep = 1", " nstep = 2", ""));
}

TEST_F(TestInputFileReader, testNotAValidKeyword)
{
    auto lineElements = vector<string>{"notAValidKeyword", "=", "1"};
    ASSERT_THROW(_inputFileReader->process(lineElements), InputFileException);
}

TEST_F(TestInputFileReader, testProcess)
{
    auto lineElements = vector<string>{"nstep", "=", "1000"};
    _inputFileReader->process(lineElements);
    EXPECT_EQ(_inputFileReader->getKeywordCount(lineElements[0]), 1);
}

TEST_F(TestInputFileReader, testRead)
{
    string filename = "data/inputFileReader/inputFile.txt";
    _inputFileReader->setFilename(filename);
    ASSERT_NO_THROW(_inputFileReader->read());
}

TEST_F(TestInputFileReader, testReadFileNotFound)
{
    string filename = "data/inputFileReader/inputFileNotFound.txt";
    _inputFileReader->setFilename(filename);
    ASSERT_THROW(_inputFileReader->read(), InputFileException);
}

TEST_F(TestInputFileReader, testReadInputFileFunction)
{
    string filename = "data/inputFileReader/inputFile.txt";
    ASSERT_NO_THROW(readInputFile(filename, _engine));
}

TEST_F(TestInputFileReader, testPostProcessRequiredFail)
{
    vector<string> keywordsRef(0);
    vector<bool>   requiredRef(0);

    ::readKeywordList("data/inputFileReader/keywordList.txt", keywordsRef, requiredRef);

    vector<size_t> requiredIndex(0);

    for (size_t i = 0; i < keywordsRef.size(); ++i)
    {
        string keyword  = keywordsRef[i];
        bool   required = requiredRef[i];

        if (required)
        {
            requiredIndex.push_back(i);
            _inputFileReader->setKeywordCount(keyword, 1);
        }
    }

    for (auto const &index : requiredIndex)
    {
        string keyword = keywordsRef[index];
        _inputFileReader->setKeywordCount(keyword, 0);
        ASSERT_THROW(_inputFileReader->postProcess(), InputFileException);
        _inputFileReader->setKeywordCount(keyword, 1);
    }
}

TEST_F(TestInputFileReader, testPostProcessCountToOftenFail)
{
    vector<string> keywordsRef(0);
    vector<bool>   requiredRef(0);

    ::readKeywordList("data/inputFileReader/keywordList.txt", keywordsRef, requiredRef);

    vector<size_t> requiredIndex(0);

    for (size_t i = 0; i < keywordsRef.size(); ++i)
    {
        string keyword  = keywordsRef[i];
        bool   required = requiredRef[i];

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
            string keyword = keywordsRef[index];
            _inputFileReader->setKeywordCount(keyword, index);
            ASSERT_THROW(_inputFileReader->postProcess(), InputFileException);
            _inputFileReader->setKeywordCount(keyword, 1);
        }
    }
}

TEST_F(TestInputFileReader, testMoldescriptorFileProcess)
{
    vector<string> keywordsRef(0);
    vector<bool>   requiredRef(0);

    ::readKeywordList("data/inputFileReader/keywordList.txt", keywordsRef, requiredRef);

    vector<size_t> requiredIndex(0);

    for (size_t i = 0; i < keywordsRef.size(); ++i)
    {
        string keyword  = keywordsRef[i];
        bool   required = requiredRef[i];

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
    InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}