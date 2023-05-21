#include <iostream>
#include <filesystem>

#ifdef WITH_MPI
#include <mpi.h>
#endif

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

int pimd_qmcf(int argc, char *argv[])
{

    // FIXME: cleanup this piece of code when knowing how to do it properly
    vector<string> arguments(argv, argv + argc);
    auto commandLineArgs = CommandLineArgs(argc, arguments);
    commandLineArgs.detectFlags();

    auto engine = Engine();

    cout << "Reading input file..." << endl;
    readInputFile(commandLineArgs.getInputFileName(), engine);

    cout << "Reading moldescriptor..." << endl;
    readMolDescriptor(engine);

    cout << "Reading rst file..." << endl;
    readRstFile(engine);

    cout << "Post processing setup..." << endl;
    postProcessSetup(engine);

    cout << "Reading guff.dat..." << endl;
    readGuffDat(engine);

    engine.calculateMomentum(engine.getSimulationBox(), engine._outputData);

    engine._logOutput->writeInitialMomentum(engine._outputData.getMomentum());
    engine._stdoutOutput->writeInitialMomentum(engine._outputData.getMomentum());

    /*
        HERE STARTS THE MAIN LOOP
    */

    if (engine._cellList.isActivated())
    {
        engine._cellList.updateCellList(engine.getSimulationBox());
    }

    // engine._jobType->calculateForces(engine.getSimulationBox(), engine._outputData);
    engine._potential->calculateForces(engine.getSimulationBox(), engine._outputData, engine._cellList);

    /*
        HERE ENDS THE MAIN LOOP
    */

    cout << "Couloumb energy: " << engine._outputData.getAverageCoulombEnergy() << endl;
    cout << "Non Couloumb energy: " << engine._outputData.getAverageNonCoulombEnergy() << endl;

    cout << "Box size: " << engine.getSimulationBox()._box.getBoxDimensions()[0] << endl;
    cout << "Box angles: " << engine.getSimulationBox()._box.getBoxAngles()[0] << endl;

    cout << "start file name: " << engine._settings.getStartFilename() << endl;

    cout << "number of steps: " << engine._timings.getNumberOfSteps() << endl;

    cout << "Moldescriptor filename: " << engine._settings.getMoldescriptorFilename() << endl;

    cout << "Water type: " << engine.getSimulationBox().getWaterType() << endl;
    cout << "Ammonia type: " << engine.getSimulationBox().getAmmoniaType() << endl;

    cout << "atom mass test: " << engine.getSimulationBox()._molecules[0].getMass(0) << endl;

    cout << "density: " << engine.getSimulationBox()._box.getDensity() << endl;

    cout << "volume: " << engine.getSimulationBox()._box.getVolume() << endl;

    return EXIT_SUCCESS;
}

// main wrapper
int main(int argc, char *argv[])
{
#ifdef WITH_MPI
    MPI_Init(&argc, &argv);
    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif
    try
    {
        int return_value = pimd_qmcf(argc, argv);

#ifdef WITH_MPI
        for (int i = 1; i < size; i++)
        {
            filesystem::remove_all("procid_pimd-qmcf_" + to_string(i));
        }
        MPI_Finalize();
#endif
        return return_value;
    }
    catch (const exception &e)
    {
        cout << "Exception: " << e.what() << endl;
#ifdef WITH_MPI
        for (int i = 1; i < size; i++)
        {
            filesystem::remove_all("procid_pimd-qmcf_" + to_string(i));
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
    }

    return EXIT_FAILURE;
}