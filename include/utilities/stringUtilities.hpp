#ifndef _STRING_UTILITIES_HPP_

#define _STRING_UTILITIES_HPP_

#include <cstddef>       // for size_t
#include <string>        // for string
#include <string_view>   // for string_view
#include <vector>        // for vector

/**
 * @brief utilities is a namespace for all utility functions
 */
namespace utilities
{
    std::string removeComments(std::string &line, const std::string_view &commentChar);

    std::vector<std::string> getLineCommands(const std::string &line, const size_t lineNumber);

    std::vector<std::string> splitString(const std::string &);

    std::string toLowerCopy(std::string);
    std::string firstLetterToUpperCaseCopy(std::string);

    bool fileExists(const std::string &);

}   // namespace utilities

#endif   // _STRING_UTILITIES_HPP_