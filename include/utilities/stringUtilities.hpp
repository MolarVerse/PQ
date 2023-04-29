#ifndef _STRING_UTILITIES_H_

#define _STRING_UTILITIES_H_

#include <string>
#include <vector>

namespace StringUtilities
{
    std::string removeComments(std::string line, std::string commentChar);
    std::vector<std::string> splitString(std::string line);
}

#endif