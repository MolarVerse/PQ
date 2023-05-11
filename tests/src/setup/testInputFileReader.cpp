#include "testInputFileReader.hpp"
#include "exceptions.hpp"

#include <gmock/gmock.h>

using namespace std;
using namespace Setup::InputFileReader;
using namespace ::testing;

TEST_F(TestInputFileReader, testAddKeyword)
{
    string filename = "data/inputFileReader/keywordList.txt";
    string line;
    ifstream inputFile(filename);

    vector<string> keywords_ref(0);
    vector<bool> required_ref(0);

    while (getline(inputFile, line))
    {
        string keyword;
        string required_string;
        bool required;

        istringstream(line) >> keyword >> required_string;
        istringstream(required_string) >> std::boolalpha >> required;

        keywords_ref.push_back(keyword);
        required_ref.push_back(bool(required));
    }

    for (size_t i = 0; i < keywords_ref.size(); i++)
    {
        string keyword = keywords_ref[i];
        bool required = required_ref[i];

        EXPECT_EQ(_inputFileReader->getKeywordCount(keyword), 0);
        EXPECT_EQ(_inputFileReader->getKeywordRequired(keyword), required);
    }
}

TEST_F(TestInputFileReader, testGetLineCommands)
{
    auto line = "nstep = 1";
    ASSERT_THROW(getLineCommands(line, 1), InputFileException);

    line = "nstep = 1;";
    ASSERT_THAT(getLineCommands(line, 1), ElementsAre("nstep = 1", ""));

    line = "nstep = 1; nstep = 2";
    ASSERT_THROW(getLineCommands(line, 1), InputFileException);

    line = "nstep = 1; nstep = 2;";
    ASSERT_THAT(getLineCommands(line, 1), ElementsAre("nstep = 1", " nstep = 2", ""));
}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}