#include "commandLineArgs.hpp"
#include "engine.hpp"

#include <iostream>
#include <string>   // for string
#include <vector>   // for vector

using namespace std;

string getExecutableName()
{
#ifdef trajectoryToCenterOfMass
    return "trajectoryToCenterOfMass";
#else
    return "analysis";
#endif
}

int main(int argc, char **argv)
{
    // like in main.cpp of pimd_qmcf not best way TODO:
    vector<string> arguments(argv, argv + argc);
    auto           commandLineArgs = CommandLineArgs(argc, arguments);
    commandLineArgs.detectFlags();

    auto executableName = getExecutableName();

    auto engine = Engine(executableName, argv[1]);

    engine.run();

    return 0;
}