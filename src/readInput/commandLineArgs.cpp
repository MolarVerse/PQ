#include "../../include/readInput/commandLineArgs.hpp"

#include "exceptions.hpp"   // for UserInputException

#include <string_view>   // for string_view

/**
 * @brief Detects flags in the command line arguments. First argument is the input file name.
 *
 * @throw UserInputException if a flag is detected (not yet implemented)
 * @throw UserInputException if no input file is specified
 */
void CommandLineArgs::detectFlags()
{
    for (const auto &arg : _argv)
        if ('-' == arg[0])
            throw customException::UserInputException("Invalid flag: " + arg + " Flags are not yet implemented.");

    if (_argc < 2)
        throw customException::UserInputException("No input file specified. Usage: pimd_qmcf <input_file>");

    _inputFileName = _argv[1];
}