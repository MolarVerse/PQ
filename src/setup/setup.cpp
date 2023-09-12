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
#include "qmSetup.hpp"
#include "qmmdEngine.hpp"
#include "resetKineticsSetup.hpp"
#include "restartFileReader.hpp"
#include "settings.hpp"
#include "simulationBoxSetup.hpp"
#include "thermostatSetup.hpp"
#include "topologyReader.hpp"

#include <iostream>

namespace engine
{
    class Engine;   // forward declaration
}

using namespace engine;
using namespace readInput;

/**
 * @brief setup the engine
 *
 * @param inputFileName
 * @param engine
 */
void setup::setupSimulation(const std::string &inputFileName, Engine &engine)
{
    readFiles(inputFileName, engine);

    std::cout << "setup engine..." << '\n';
    setupEngine(engine);

    // needs setup of engine before reading guff.dat
    std::cout << "Reading guff.dat..." << '\n';
    guffdat::readGuffDat(engine);

    std::cout << "Setup complete!" << '\n';
}

/**
 * @brief reads all the files needed for the simulation
 *
 * @param inputFileName
 * @param engine
 */
void setup::readFiles(const std::string &inputFileName, Engine &engine)
{
    std::cout << "Reading input file..." << '\n';
    readInputFile(inputFileName, engine);

    std::cout << "Reading moldescriptor..." << '\n';
    molDescriptor::readMolDescriptor(engine);

    std::cout << "Reading rst file..." << '\n';
    restartFile::readRestartFile(engine);

    std::cout << "Reading topology file..." << '\n';
    topology::readTopologyFile(engine);

    std::cout << "Reading parameter file..." << '\n';
    parameterFile::readParameterFile(engine);

    std::cout << "Reading intra non bonded file..." << '\n';
    readInput::intraNonBonded::readIntraNonBondedFile(engine);
}

/**
 * @brief setup the engine
 *
 * @param engine
 */
void setup::setupEngine(Engine &engine)
{
    if (settings::Settings::getIsQM())
    {
        std::cout << "setup QM" << '\n';
        setupQM(dynamic_cast<engine::QMMDEngine &>(engine));
    }

    std::cout << "setup simulation box" << '\n';
    setupSimulationBox(engine);

    std::cout << "setup cell list" << '\n';
    setupCellList(engine);

    std::cout << "setup thermostat" << '\n';
    setupThermostat(engine);

    std::cout << "setup manostat" << '\n';
    setupManostat(engine);

    std::cout << "setup reset kinetics" << '\n';
    setupResetKinetics(engine);

    if (settings::Settings::getIsMM())
    {
        std::cout << "setup potential" << '\n';
        setupPotential(engine);   // has to be after simulationBox setup due to coulomb radius cutoff

        std::cout << "intra non bonded" << '\n';
        setupIntraNonBonded(engine);

        std::cout << "setup force field" << '\n';
        setupForceField(engine);
    }

    std::cout << "setup constraints" << '\n';
    setupConstraints(engine);
}