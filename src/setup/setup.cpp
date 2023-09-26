#include "setup.hpp"

#include "celllistSetup.hpp"          // for setupCellList
#include "constraintsSetup.hpp"       // for setupConstraints
#include "engine.hpp"                 // for Engine
#include "forceFieldSetup.hpp"        // for setupForceField
#include "guffDatReader.hpp"          // for readGuffDat, readInput
#include "inputFileReader.hpp"        // for readInputFile
#include "intraNonBondedReader.hpp"   // for readIntraNonBondedFile
#include "intraNonBondedSetup.hpp"    // for setupIntraNonBonded
#include "manostatSetup.hpp"          // for setupManostat
#include "moldescriptorReader.hpp"    // for readMolDescriptor
#include "outputFilesSetup.hpp"       // for setupOutputFiles
#include "parameterFileReader.hpp"    // for readParameterFile
#include "potentialSetup.hpp"         // for setupPotential
#include "qmSetup.hpp"                // for setupQM
#include "qmmdEngine.hpp"             // for QMMDEngine
#include "resetKineticsSetup.hpp"     // for setupResetKinetics
#include "restartFileReader.hpp"      // for readRestartFile
#include "ringPolymerEngine.hpp"      // for RingPolymerEngine
#include "ringPolymerSetup.hpp"       // for setupRingPolymer
#include "settings.hpp"               // for Settings
#include "simulationBoxSetup.hpp"     // for setupSimulationBox
#include "thermostatSetup.hpp"        // for setupThermostat
#include "topologyReader.hpp"         // for readTopologyFile

#include <iostream>   // for operator<<, basic_ostream, cout

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
    std::cout << "Reading input file..." << '\n';
    readInputFile(inputFileName, engine);

    std::cout << "setup output files..." << '\n';
    setupOutputFiles(engine);

    readFiles(engine);

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
void setup::readFiles(Engine &engine)
{
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
    if (settings::Settings::isQMActivated())
    {
        std::cout << "setup QM" << '\n';
        setupQM(dynamic_cast<engine::QMMDEngine &>(engine));
    }

    std::cout << "setup reset kinetics" << '\n';
    setupResetKinetics(engine);

    std::cout << "setup simulation box" << '\n';
    setupSimulationBox(engine);

    std::cout << "setup cell list" << '\n';
    setupCellList(engine);

    std::cout << "setup thermostat" << '\n';
    setupThermostat(engine);

    std::cout << "setup manostat" << '\n';
    setupManostat(engine);

    if (settings::Settings::isMMActivated())
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

    if (settings::Settings::isRingPolymerMDActivated())
    {
        std::cout << "setup ring polymer" << '\n';
        setupRingPolymer(dynamic_cast<engine::RingPolymerEngine &>(engine));
    }
}