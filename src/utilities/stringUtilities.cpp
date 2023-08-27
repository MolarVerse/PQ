#include "stringUtilities.hpp"

#include "exceptions.hpp"   // for InputFil...

#include <algorithm>                                          // for __for_each_fn
#include <boost/algorithm/string/classification.hpp>          // for is_any_of
#include <boost/algorithm/string/detail/classification.hpp>   // for is_any_ofF
#include <boost/algorithm/string/split.hpp>                   // for split
#include <boost/iterator/iterator_facade.hpp>                 // for operator!=
#include <boost/type_index/type_index_facade.hpp>             // for operator==
#include <cctype>                                             // for isspace
#include <format>                                             // for format
#include <fstream>                                            // for basic_istream, ifstream
#include <functional>                                         // for identity
#include <sstream>                                            // for sstream
#include <string>                                             // for string
#include <vector>                                             // for vector

/**
 * @brief Removes comments from a line
 *
 * @details the comment char can be chosen freely - in the program ';' is used
 *
 * @param line
 * @param commentChar
 * @return std::string
 */
std::string utilities::removeComments(std::string &line, const std::string_view &commentChar)
{
    if (const auto commentPos = line.find(commentChar); commentPos != std::string::npos)
        line = line.substr(0, commentPos);
    return line;
}

/**
 * @brief get commands from a line
 *
 * @note split commands at every semicolon
 *
 * @param line
 * @return std::vector<std::string>
 *
 * @throw InputFileException if line does not end with a semicolon
 */
std::vector<std::string> utilities::getLineCommands(const std::string &line, const size_t lineNumber)
{

    for (auto i = static_cast<int>(line.size() - 1); i >= 0; --i)
    {
        if (';' == line[static_cast<size_t>(i)])
            break;
        else if (!bool(::isspace(line[static_cast<size_t>(i)])))
            throw customException::InputFileException(std::format("Missing semicolon in input file at line {}", lineNumber));
    }

    std::vector<std::string> lineCommands;
    boost::split(lineCommands, line, boost::is_any_of(";"));

    return lineCommands;
}

/**
 * @brief Splits a string into a vector of strings at every whitespace
 *
 * @param line
 * @return std::vector<std::string>
 */
std::vector<std::string> utilities::splitString(const std::string &line)
{
    std::string              word;
    std::vector<std::string> lineElements = {};

    std::stringstream ss(line);

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
std::string utilities::toLowerCopy(std::string myString)
{
    std::ranges::for_each(myString, [](char &c) { c = char(::tolower(c)); });
    return myString;
}

/**
 * @brief checks if a file exists and can be opened
 *
 * @param filename
 * @return true if file exists and can be opened
 * @return false if file does not exist or cannot be opened
 */
bool utilities::fileExists(const std::string &filename)
{
    std::ifstream file(filename);
    return file.good();
}