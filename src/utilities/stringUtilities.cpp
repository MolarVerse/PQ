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
string StringUtilities::removeComments(string &line, const string_view &commentChar)
{
    if (const auto commentPos = line.find(commentChar); commentPos != string::npos)
        line = line.substr(0, commentPos);
    return line;
}

/**
 * @brief Splits a string into a vector of strings
 *
 * @param line
 * @return vector<string>
 *
 * TODO: merge splitstring functions
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
vector<string> StringUtilities::getLineCommands(const string &line, const size_t lineNumber)
{

    for (auto i = static_cast<int>(line.size() - 1); i >= 0; --i)
    {
        if (line[static_cast<size_t>(i)] == ';')
            break;
        else if (!isspace(line[static_cast<size_t>(i)]))
            throw InputFileException("Missing semicolon in input file at line " + to_string(lineNumber));
        else
        {
            // dummy
        }
    }

    vector<string> lineCommands;
    boost::split(lineCommands, line, boost::is_any_of(";"));

    return lineCommands;
}