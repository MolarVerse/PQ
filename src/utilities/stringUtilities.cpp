#include <string>
#include <sstream>
#include <vector>

#include "stringUtilities.hpp"

using namespace std;

string StringUtilities::removeComments(string line, string commentChar)
{
    auto commentPos = line.find(commentChar);
    if (commentPos != string::npos)
        line = line.substr(0, commentPos);
    return line;
}

vector<string> StringUtilities::splitString(string line)
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