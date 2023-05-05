#include <stdexcept>

#include <fstream>

#include "output.hpp"

using namespace std;

void Output::setFilename(string_view filename)
{
    if (filename.empty())
        throw invalid_argument("Filename cannot be empty");

    ifstream fp(string(filename).c_str());
    if (fp.good())
        throw invalid_argument("File already exists - filename = " + string(filename));

    _filename = filename;
}

/**
 * @brief Sets the output frequency of the simulation
 *
 * @param outputFreq
 *
 * @throw range_error if output frequency is negative
 *
 * @note
 *
 *  if output frequency is 0, it is set to INT32_MAX
 *  in order to avoid division by 0 in the output
 *
 */
void Output::setOutputFreq(int outputFreq)
{
    if (outputFreq < 0)
        throw range_error("Output frequency must be positive - output frequency = " + to_string(outputFreq));

    if (outputFreq == 0)
        _outputFreq = INT32_MAX;
    else
        _outputFreq = outputFreq;
}