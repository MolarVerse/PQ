#include "stringUtilities.hpp"

#include "exceptions.hpp"

#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;
using namespace customException;

/**
 * @brief Removes comments from a line
 *
 * @param line
 * @param commentChar
 * @return string
 */
string StringUtilities::removeComments(string &line, const string_view &commentChar)
{
    if (const auto commentPos = line.find(commentChar); commentPos != string::npos) line = line.substr(0, commentPos);
    return line;
}

/**
 * @brief get commands from a line
 *
 * @note split commands at every semicolon
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
        else if (!::isspace(line[static_cast<size_t>(i)]))
            throw InputFileException("Missing semicolon in input file at line " + to_string(lineNumber));
    }

    vector<string> lineCommands;
    boost::split(lineCommands, line, boost::is_any_of(";"));

    return lineCommands;
}

/**
 * @brief Splits a string into a vector of strings at every whitespace
 *
 * @param line
 * @return vector<string>
 */
vector<string> StringUtilities::splitString(const string &line)
{
    string         word;
    vector<string> lineElements = {};

    stringstream ss(line);

    while (ss >> word)
        lineElements.push_back(word);

    return lineElements;
}

/**
 * @brief returns a copy of a string all lower case
 *
 * @param myString
 * @return string
 */
string StringUtilities::toLowerCopy(string myString)
{
    ranges::for_each(myString, [](auto &c) { c = ::tolower(c); });
    return myString;
}

/**
 * @brief checks if a file exists and can be opened
 *
 * @param filename
 * @return true if file exists and can be opened
 * @return false if file does not exist or cannot be opened
 */
bool StringUtilities::fileExists(const string &filename)
{
    ifstream file(filename);
    return file.good();
}