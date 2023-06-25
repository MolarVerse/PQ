#include "commandLineArgs.hpp"

#include "exceptions.hpp"

#include <boost/algorithm/string/predicate.hpp>
#include <boost/lexical_cast.hpp>
#include <iostream>

using namespace std;
using namespace customException;

CommandLineArgs::CommandLineArgs(const int argc, const vector<string> &argv) : _argc(argc), _argv(argv) {}

/**
 * @brief Detects flags in the command line arguments.
 *
 * @throw UserInputException if a flag is detected (not yet implemented)
 * @throw UserInputException if no input file is specified
 */
void CommandLineArgs::detectFlags()
{
    for (const auto &arg : _argv)
        if (boost::starts_with(arg, "-")) throw UserInputException("Invalid flag: " + arg + " Flags are not yet implemented.");

    if (_argc < 2) throw UserInputException("No input file specified. Usage: pimd_qmcf <input_file>");

    _inputFileName = _argv[1];
}