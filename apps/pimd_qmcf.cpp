#include <iostream>

#include "rstFileReader.hpp"
#include "simulationBox.hpp"
#include "commandLineArgs.hpp"
#include "inputFileReader.hpp"
#include "settings.hpp"

using namespace std;

int main(int argc, char *argv[])
{
    // FIXME: cleanup this piece of code when knowing how to do it properly
    vector<string> arguments(argv, argv + argc);
    auto commandLineArgs = CommandLineArgs(argc, arguments);
    commandLineArgs.detectFlags();

    auto settings = Settings();
    readInputFile(commandLineArgs.getInputFileName(), settings);

    auto simulationBox = read_rst("h2o-qmcf.rst", settings);

    cout << "Step count: " << settings._timings.getStepCount() << endl;
    cout << "Timestep: " << settings._timings.getTimestep() << endl;
    cout << "Job type: " << settings._jobType.getJobType() << endl;

    cout << commandLineArgs.getInputFileName() << endl;

    return 0;
}