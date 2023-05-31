#include <stdexcept>
#include <filesystem>

#include <fstream>

#include "output.hpp"
#include "exceptions.hpp"

#ifdef WITH_MPI
#include <mpi.h>
#endif

using namespace std;

/**
 * @brief Sets the filename of the output file
 *
 * @param filename
 *
 * @throw InputFileException if filename is empty
 */
void Output::setFilename(const string_view &filename)
{
#ifdef WITH_MPI
    int rank;
    int procId;
    MPI_Comm_rank(MPI_COMM_WORLD, &procId);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
        _filename = filename;
    }
    else
    {
        filesystem::create_directory("procid_pimd-qmcf_" + to_string(procId));
        _filename = filename;
        _filename = "procid_pimd-qmcf_" + to_string(procId) + "/" + _filename;
    }
#else

    _filename = filename;
#endif

    if (_filename.empty())
    {
        throw InputFileException("Filename cannot be empty");
    }

    if (const ifstream fp(_filename.c_str()); fp.good())
    {
        throw InputFileException("File already exists - filename = " + string(_filename));
    }

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
void Output::setOutputFreq(const int outputFreq)
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
string Output::initialMomentumMessage(const double momentum) const
{
    return "Initial momentum = " + to_string(momentum) + " Angstrom * amu / fs";
}

/**
 * @brief write initial momentum to log file
 *
 * @param momentum
 */
void LogOutput::writeInitialMomentum(const double momentum)
{
    _fp << endl;
    _fp << initialMomentumMessage(momentum) << endl;
}

/**
 * @brief write initial momentum to stdout
 *
 * @param momentum
 */
void StdoutOutput::writeInitialMomentum(const double momentum) const
{
    cout << endl;
    cout << initialMomentumMessage(momentum) << endl;
}

/**
 * @brief write warning message to log file if Berendsen thermostat is set but no relaxation time is given
 *
 */
void LogOutput::writeRelaxationTimeThermostatWarning()
{
    _fp << "WARNING: Berendsen thermostat set but no relaxation time given. Using default value of 0.1ps." << endl;
}

/**
 * @brief write warning message to stdout if Berendsen thermostat is set but no relaxation time is given
 *
 */
void StdoutOutput::writeRelaxationTimeThermostatWarning() const
{
    try
    {
        throw UserInputExceptionWarning("Berendsen thermostat set but no relaxation time given. Using default value of 0.1ps.");
    }
    catch (const UserInputExceptionWarning &e)
    {
        cout << e.what() << endl
             << endl;
    }
}