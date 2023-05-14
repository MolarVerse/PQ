#include <string>
#include <sstream>
#include <vector>
#include <boost/algorithm/string.hpp>

#include "stringUtilities.hpp"
#include "exceptions.hpp"

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

/**
 * @brief Splits a string into a vector of strings
 *
 * @param line
 * @return vector<string>
 */
void StringUtilities::splitString2(const string &line, vector<string> &lineElements)
{
    string word;
    lineElements.clear();

    stringstream ss(line);

    while (ss >> word)
    {
        lineElements.push_back(word);
    }
}

/**
 * @brief get commands from a line
 *
 * @param line
 * @return vector<string>
 *
 * @throw InputFileException if line does not end with a semicolon
 */
vector<string> StringUtilities::getLineCommands(const string &line, int _lineNumber)
{

    for (int i = int(line.size()) - 1; i >= 0; i--)
    {
        if (line[i] == ';')
            break;
        else if (!isspace(line[i]))
            throw InputFileException("Missing semicolon in input file at line " + to_string(_lineNumber));
    }

    vector<string> lineCommands;
    boost::split(lineCommands, line, boost::is_any_of(";"));

    return lineCommands;
}