#include <iostream>

#include "rstFileReader.hpp"
#include "simulationBox.hpp"
#include "commandLineArgs.hpp"
#include "inputFileReader.hpp"
#include "output.hpp"
#include "engine.hpp"
#include "moldescriptorReader.hpp"

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

    readMolDescriptor(engine);

    read_rst(engine);

    cout << "Box size: " << engine._simulationBox._box.getBoxDimensions()[0] << endl;
    cout << "Box size: " << engine._simulationBox._box.getBoxAngles()[1] << endl;

    cout << engine._settings.getStartFilename() << endl;

    cout << engine._jobType.getJobType() << endl;

    cout << engine._settings._timings.getNumberOfSteps() << endl;

    for (auto output : engine._output)
    {
        cout << output.getFilename() << endl;
    }

    cout << "Moldescriptor filename: " << engine._settings.getMoldescriptorFilename() << endl;

    cout << "Water type: " << engine._simulationBox.getWaterType() << endl;
    cout << "Ammonia type: " << engine._simulationBox.getAmmoniaType() << endl;

    for (auto molecule : engine._simulationBox._molecules)
    {
        for (int i = 0; i < molecule.getNumberOfAtoms(); i++)
        {
            cout << "x:   " << molecule.getAtomPosition(i)[0] << endl;
            cout << "y:   " << molecule.getAtomPosition(i)[1] << endl;
            cout << "z:   " << molecule.getAtomPosition(i)[2] << endl;
        }
        cout << endl;
    }

    return 0;
}