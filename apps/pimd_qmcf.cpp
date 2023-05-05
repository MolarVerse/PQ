#include <iostream>

#include "rstFileReader.hpp"
#include "simulationBox.hpp"
#include "commandLineArgs.hpp"
#include "inputFileReader.hpp"
#include "output.hpp"
#include "engine.hpp"

#include "initStatic.hpp"

using namespace std;

int main(int argc, char *argv[])
{
    // FIXME: cleanup this piece of code when knowing how to do it properly
    vector<string> arguments(argv, argv + argc);
    auto commandLineArgs = CommandLineArgs(argc, arguments);
    commandLineArgs.detectFlags();

    auto engine = Engine();
    readInputFile(commandLineArgs.getInputFileName(), engine);

    auto simulationBox = read_rst(engine._settings.getStartFilename(), engine._settings);

    return 0;
}