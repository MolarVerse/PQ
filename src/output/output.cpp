#include "output.hpp"

#include "exceptions.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>

#ifdef WITH_MPI
#include <mpi.h>
#endif

using namespace std;
using namespace customException;
using namespace output;

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

    if (_filename.empty()) throw InputFileException("Filename cannot be empty");

    if (const ifstream fp(_filename.c_str()); fp.good())
        throw InputFileException("File already exists - filename = " + string(_filename));

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
 * TODO: check if output frequency is positive in parser file
 *
 */
void Output::setOutputFrequency(const size_t outputFreq)
{
    // if (outputFreq < 0)
    //     throw InputFileException("Output frequency must be positive - output frequency = " + to_string(outputFreq));

    if (outputFreq == 0)
        _outputFrequency = INT64_MAX;
    else
        _outputFrequency = outputFreq;
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

    if (!_fp.is_open()) throw InputFileException("Could not open file - filename = " + _filename);
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
