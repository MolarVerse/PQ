#ifndef _STRING_UTILITIES_HPP_

#define _STRING_UTILITIES_HPP_

#include <string>
#include <vector>

/**
 * @brief utilities is a namespace for all utility functions
 */
namespace utilities
{
    std::string removeComments(std::string &, const std::string_view &);

    std::vector<std::string> getLineCommands(const std::string &, const size_t);
    std::vector<std::string> splitString(const std::string &);

    std::string toLowerCopy(std::string);

    bool fileExists(const std::string &);

}   // namespace utilities

#endif   // _STRING_UTILITIES_HPP_