#include <iostream>

#include "rstFileReader.hpp"
#include "simulationBox.hpp"
#include "commandLineArgs.hpp"
#include "inputFileReader.hpp"
#include "output.hpp"
#include "engine.hpp"
#include "moldescriptorReader.hpp"
#include "postProcessSetup.hpp"

using namespace std;

int main(int argc, char *argv[])
{
    // FIXME: cleanup this piece of code when knowing how to do it properly
    vector<string> arguments(argv, argv + argc);
    auto commandLineArgs = CommandLineArgs(argc, arguments);
    commandLineArgs.detectFlags();

    auto engine = Engine();
    readInputFile(commandLineArgs.getInputFileName(), engine);

    readMolDescriptor(engine);

    readRstFile(engine);

    postProcessSetup(engine);

    cout << "Box size: " << engine.getSimulationBox()._box.getBoxDimensions()[0] << endl;
    cout << "Box size: " << engine.getSimulationBox()._box.getBoxAngles()[1] << endl;

    cout << engine._settings.getStartFilename() << endl;

    cout << engine._jobType.getJobType() << endl;

    cout << engine._settings._timings.getNumberOfSteps() << endl;

    cout << engine._logOutput->getFilename() << endl;
    cout << engine._logOutput->getOutputFreq() << endl;

    cout << "Moldescriptor filename: " << engine._settings.getMoldescriptorFilename() << endl;

    cout << "Water type: " << engine.getSimulationBox().getWaterType() << endl;
    cout << "Ammonia type: " << engine.getSimulationBox().getAmmoniaType() << endl;

    cout << "atom mass test: " << engine.getSimulationBox()._molecules[0].getMass(0) << endl;

    // for (auto molecule : engine.getSimulationBox()._molecules)
    // {
    //     for (int i = 0; i < molecule.getNumberOfAtoms(); i++)
    //     {
    //         cout << "x:   " << molecule.getAtomPosition(i)[0] << endl;
    //         cout << "y:   " << molecule.getAtomPosition(i)[1] << endl;
    //         cout << "z:   " << molecule.getAtomPosition(i)[2] << endl;
    //     }
    //     cout << endl;
    // }

    cout << "density " << engine.getSimulationBox()._box.getDensity() << endl;

    return 0;
}