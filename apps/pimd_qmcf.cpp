#include <iostream>

#include "rstFileReader.hpp"
#include "simulationBox.hpp"
#include "commandLineArgs.hpp"
#include "inputFileReader.hpp"
#include "output.hpp"
#include "engine.hpp"
#include "moldescriptorReader.hpp"
#include "postProcessSetup.hpp"
#include "guffDatReader.hpp"
#include "celllist.hpp"

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

    readGuffDat(engine);

    engine.calculateMomentum(engine.getSimulationBox(), engine._outputData);

    engine._logOutput->writeInitialMomentum(engine._outputData.getMomentum());
    engine._stdoutOutput->writeInitialMomentum(engine._outputData.getMomentum());

    engine._cellList.setup(engine.getSimulationBox());

    engine._cellList.updateCellList(engine.getSimulationBox());

    int numberOfMolecules = 0;
    for (auto &cell : engine._cellList.getCells())
    {
        numberOfMolecules += cell.getNumberOfMolecules();
        cout << cell.getNumberOfMolecules() << endl;
    }

    cout << "Number of molecules: " << numberOfMolecules << endl;

    engine._jobType->calculateForces(engine.getSimulationBox(), engine._outputData);

    cout << engine._outputData.getAverageCoulombEnergy() << endl;
    cout << engine._outputData.getAverageNonCoulombEnergy() << endl;

    cout << "Box size: " << engine.getSimulationBox()._box.getBoxDimensions()[0] << endl;
    cout << "Box size: " << engine.getSimulationBox()._box.getBoxAngles()[1] << endl;

    cout << engine._settings.getStartFilename() << endl;

    cout << engine._jobType->getJobType() << endl;

    cout << engine._settings._timings.getNumberOfSteps() << endl;

    cout << engine._logOutput->getFilename() << endl;
    cout << engine._logOutput->getOutputFreq() << endl;

    cout << "Moldescriptor filename: " << engine._settings.getMoldescriptorFilename() << endl;

    cout << "Water type: " << engine.getSimulationBox().getWaterType() << endl;
    cout << "Ammonia type: " << engine.getSimulationBox().getAmmoniaType() << endl;

    cout << "atom mass test: " << engine.getSimulationBox()._molecules[0].getMass(0) << endl;

    cout << "density " << engine.getSimulationBox()._box.getDensity() << endl;

    cout << engine.getSimulationBox().getGuffCoefficients(1, 1, 0, 0)[0] << endl;
    cout << engine.getSimulationBox().getGuffCoefficients(1, 1, 0, 1)[0] << endl;
    cout << engine.getSimulationBox().getGuffCoefficients(1, 1, 1, 0)[0] << endl;
    cout << engine.getSimulationBox().getGuffCoefficients(1, 1, 1, 1)[0] << endl;

    cout << engine.getSimulationBox().getRncCutOff(1, 1, 0, 0) << endl;
    cout << engine.getSimulationBox().getRncCutOff(1, 1, 0, 1) << endl;
    cout << engine.getSimulationBox().getRncCutOff(1, 1, 1, 0) << endl;
    cout << engine.getSimulationBox().getRncCutOff(1, 1, 1, 1) << endl;

    return 0;
}