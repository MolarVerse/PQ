#include "setup.hpp"

#include "celllistSetup.hpp"
#include "constraintsSetup.hpp"
#include "forceFieldSetup.hpp"
#include "guffDatReader.hpp"
#include "inputFileReader.hpp"
#include "integratorSetup.hpp"
#include "intraNonBondedReader.hpp"
#include "intraNonBondedSetup.hpp"
#include "manostatSetup.hpp"
#include "moldescriptorReader.hpp"
#include "parameterFileReader.hpp"
#include "potentialSetup.hpp"
#include "resetKineticsSetup.hpp"
#include "rstFileReader.hpp"
#include "simulationBoxSetup.hpp"
#include "thermostatSetup.hpp"
#include "topologyReader.hpp"

#include <iostream>

namespace engine
{
    class Engine;
}   // namespace engine

using namespace std;
using namespace engine;
using namespace readInput;

/**
 * @brief setup the engine
 *
 * @param inputFilename
 * @param engine
 */
void setup::setupSimulation(const string &inputFilename, Engine &engine)
{
    readFiles(inputFilename, engine);

    cout << "setup engine..." << '\n';
    setupEngine(engine);

    // needs setup of engine before reading guff.dat
    cout << "Reading guff.dat..." << '\n';
    readGuffDat(engine);

    cout << "Setup complete!" << '\n';
}

/**
 * @brief reads all the files needed for the simulation
 *
 * @param inputFilename
 * @param engine
 */
void setup::readFiles(const string &inputFilename, Engine &engine)
{
    cout << "Reading input file..." << '\n';
    readInputFile(inputFilename, engine);

    cout << "Reading moldescriptor..." << '\n';
    readMolDescriptor(engine);

    cout << "Reading rst file..." << '\n';
    readRstFile(engine);

    cout << "Reading topology file..." << '\n';
    topology::readTopologyFile(engine);

    cout << "Reading parameter file..." << '\n';
    parameterFile::readParameterFile(engine);

    cout << "Reading intra non bonded file..." << '\n';
    readIntraNonBondedFile(engine);
}

/**
 * @brief setup the engine
 *
 * @param engine
 */
void setup::setupEngine(Engine &engine)
{
    setupSimulationBox(engine);
    setupCellList(engine);
    setupThermostat(engine);
    setupManostat(engine);
    setupResetKinetics(engine);
    setupPotential(engine);   // has to be after simulationBox setup due to coulomb radius cutoff
    setupIntegrator(engine);
    setupConstraints(engine);

    setupIntraNonBonded(engine);
    setupForceField(engine);
}