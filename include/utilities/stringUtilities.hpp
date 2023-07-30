#ifndef _STRING_UTILITIES_HPP_

#define _STRING_UTILITIES_HPP_

#include <string>
#include <vector>

/**
 * @brief StringUtilities is a namespace for string utilities
 */
namespace StringUtilities
{
    std::string removeComments(std::string &, const std::string_view &);

    std::vector<std::string> getLineCommands(const std::string &, const size_t);
    std::vector<std::string> splitString(const std::string &);

    std::string to_lower_copy(std::string);

}   // namespace StringUtilities

#endif   // _STRING_UTILITIES_HPP_