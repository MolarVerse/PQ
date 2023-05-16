#ifndef _STRING_UTILITIES_H_

#define _STRING_UTILITIES_H_

#include <string>
#include <vector>

/**
 * @brief StringUtilities is a namespace for string utilities
 *
 */
namespace StringUtilities
{
    std::string removeComments(std::string &, std::string_view);
    std::vector<std::string> getLineCommands(const std::string &, int);

    std::vector<std::string> splitString(const std::string &);
    void splitString2(const std::string &, std::vector<std::string> &);
}

#endif