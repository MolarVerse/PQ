#include <stdexcept>

#include <fstream>

#include "output.hpp"
#include "exceptions.hpp"

using namespace std;

/**
 * @brief Sets the filename of the output file
 *
 * @param filename
 *
 * @throw InputFileException if filename is empty
 */
void Output::setFilename(string_view filename)
{
    if (filename.empty())
        throw InputFileException("Filename cannot be empty");

    if (ifstream fp(string(filename).c_str()); fp.good())
        throw InputFileException("File already exists - filename = " + string(filename));

    _filename = filename;

    openFile();
}

/**
 * @brief Sets the output frequency of the simulation
 *
 * @param outputFreq
 *
 * @throw InputFileException if output frequency is negative
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
        throw InputFileException("Output frequency must be positive - output frequency = " + to_string(outputFreq));

    if (outputFreq == 0)
        _outputFreq = INT32_MAX;
    else
        _outputFreq = outputFreq;
}

/**
 * @brief Opens the output file
 *
 * @throw InputFileException if file cannot be opened
 *
 */
void Output::openFile()
{
    _fp.open(_filename);

    if (!_fp.is_open())
        throw InputFileException("Could not open file - filename = " + _filename);
}

/**
 * @brief write a warning message to the log file if density and box dimensions are set
 *
 */
void LogOutput::writeDensityWarning()
{
    _fp << "WARNING: Density and box dimensions set. Density will be ignored." << endl;
}

/**
 * @brief write a warning message to the stdout if density and box dimensions are set
 *
 */
void StdoutOutput::writeDensityWarning() const
{
    try
    {
        throw UserInputExceptionWarning("Density and box dimensions set. Density will be ignored.");
    }
    catch (const UserInputExceptionWarning &e)
    {
        cout << e.what() << endl
             << endl;
    }
}

/**
 * @brief construct general initial momentum message
 *
 * @param momentum
 * @return string
 */
string Output::initialMomentumMessage(double momentum) const
{
    return "Initial momentum = " + to_string(momentum) + " Angstrom * amu / fs";
}

/**
 * @brief write initial momentum to log file
 *
 * @param momentum
 */
void LogOutput::writeInitialMomentum(double momentum)
{
    _fp << endl;
    _fp << initialMomentumMessage(momentum) << endl;
}

/**
 * @brief write initial momentum to stdout
 *
 * @param momentum
 */
void StdoutOutput::writeInitialMomentum(double momentum) const
{
    cout << endl;
    cout << initialMomentumMessage(momentum) << endl;
}
