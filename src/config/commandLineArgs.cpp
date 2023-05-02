#include <stdexcept>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/lexical_cast.hpp>
#include <iostream>

#include "commandLineArgs.hpp"

using namespace std;

CommandLineArgs::CommandLineArgs(int argc, const vector<string> &argv) : _argc(argc), _argv(argv) {}

/**
 * @brief Detects flags in the command line arguments.
 * 
 */
void CommandLineArgs::detectFlags()
{
    for (const auto &arg: _argv){
        if(boost::starts_with(arg, "-")){
            throw invalid_argument("Invalid flag: " + arg + " Flags are not yet implemented.");
        }
    }

    if(_argc < 2){
        throw invalid_argument("No input file specified. Usage: pimd_qmcf <input_file>");
    }

    _inputFileName = _argv[1];
}