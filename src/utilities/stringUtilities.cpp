#include <string>

#include "stringUtilities.hpp"

using namespace std;

string StringUtilities::removeComments(string line, string commentChar)
{
    auto commentPos = line.find(commentChar);
    if(commentPos != string::npos) line = line.substr(0, commentPos);
    return line;
}