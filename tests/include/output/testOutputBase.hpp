#include <fstream>
#include <sstream>

/**
 * @brief Reads the entire contents of a file into a string.
 *
 * @param path The path to the file to read.
 * @return A string containing the contents of the file.
 */
inline std::string slurp(const std::string &path)
{
    std::ifstream     fileStreamIn(path);
    std::stringstream stringStream;
    stringStream << fileStreamIn.rdbuf();
    return stringStream.str();
}
