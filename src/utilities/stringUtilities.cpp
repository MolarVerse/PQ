#include <string>
#include <sstream>
#include <vector>

#include "stringUtilities.hpp"

using namespace std;

/**
 * @brief Removes comments from a line
 *
 * @param line
 * @param commentChar
 * @return string
 */
string StringUtilities::removeComments(string &line, string_view commentChar)
{
    if (auto commentPos = line.find(commentChar); commentPos != string::npos)
        line = line.substr(0, commentPos);
    return line;
}

/**
 * @brief Splits a string into a vector of strings
 *
 * @param line
 * @return vector<string>
 */
vector<string> StringUtilities::splitString(const string &line)
{
    string word;
    vector<string> lineElements = {};

    stringstream ss(line);

    while (ss >> word)
    {
        lineElements.push_back(word);
    }

    return lineElements;
}