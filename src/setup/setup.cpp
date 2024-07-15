/*****************************************************************************
<GPL_HEADER>

    PQ
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

#include "setup.hpp"

#include <iostream>   // for operator<<, basic_ostream, cout

#include "celllistSetup.hpp"          // for setupCellList
#include "constraintsSetup.hpp"       // for setupConstraints
#include "engine.hpp"                 // for Engine
#include "forceFieldSetup.hpp"        // for setupForceField
#include "guffDatReader.hpp"          // for readGuffDat, readInput
#include "hybridSetup.hpp"            // for setupQMMM
#include "inputFileReader.hpp"        // for readInputFile
#include "intraNonBondedReader.hpp"   // for readIntraNonBondedFile
#include "intraNonBondedSetup.hpp"    // for setupIntraNonBonded
#include "kokkosSetup.hpp"            // for setupKokkos
#include "manostatSetup.hpp"          // for setupManostat
#include "moldescriptorReader.hpp"    // for readMolDescriptor
#include "optimizerSetup.hpp"         // for setupOptimizer
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
#include "timer.hpp"                  // for Timings
#include "timingsSettings.hpp"        // for TimingsSettings
#include "topologyReader.hpp"         // for readTopologyFile

using namespace engine;
using namespace input;
using namespace timings;
using namespace settings;
using namespace customException;
using namespace guffdat;
using namespace molDescriptor;
using namespace restartFile;
using namespace topology;
using namespace parameterFile;
using namespace input::intraNonBondedReader;
using namespace setup::simulationBox;
using namespace setup::resetKinetics;

/**
 * @brief setup the engine
 *
 * @param inputFileName
 * @param engine
 */
void setup::setupRequestedJob(const std::string &inputFileName, Engine &engine)
{
    auto simulationTimer = Timer("Simulation");
    auto setupTimer      = Timer("Setup");

    startSetup(simulationTimer, setupTimer, engine);

    readInputFile(inputFileName, engine);

    if (!TimingsSettings::isTimeStepSet())
        if (Settings::isMDJobType())
            throw UserInputException(std::format(
                "Molecular Dynamics job type {} selected. Please set the time "
                "step in the input file.",
                string(Settings::getJobtype())
            ));

    setupOutputFiles(engine);

    readFiles(engine);

    setupEngine(engine);

    // needs setup of engine before reading guff.dat
    readGuffDat(engine);

#ifdef WITH_KOKKOS
    setupKokkos(engine);
#endif

    endSetup(simulationTimer, setupTimer, engine);
}

/**
 * @brief start the setup
 *
 * @param engine
 */
void setup::startSetup(
    Timer  &simulationTimer,
    Timer  &setupTimer,
    Engine &engine
)
{
    simulationTimer.startTimingsSection();
    setupTimer.startTimingsSection("TotalSetup");

    engine.getStdoutOutput().writeHeader();
}

/**
 * @brief end the setup
 *
 * @param engine
 */
void setup::endSetup(
    const Timer &simulationTimer,
    Timer       &setupTimer,
    Engine      &engine
)
{
    engine.getStdoutOutput().writeSetupCompleted();
    engine.getLogOutput().writeSetupCompleted();

    setupTimer.stopTimingsSection("TotalSetup");
    engine.getTimer().addSimulationTimer(simulationTimer);
    engine.addTimer(setupTimer);
}

/**
 * @brief reads all the files needed for the simulation
 *
 * @param inputFileName
 * @param engine
 */
void setup::readFiles(Engine &engine)
{
    readMolDescriptor(engine);

    readRestartFile(engine);

    readTopologyFile(engine);

    readParameterFile(engine);

    readIntraNonBondedFile(engine);
}

/**
 * @brief setup the engine
 *
 * @param engine
 */
void setup::setupEngine(Engine &engine)
{
    if (Settings::isQMActivated())
        setupQM(engine);

    if (Settings::isMDJobType())
        setupResetKinetics(engine);

    setupSimulationBox(engine);

    setupCellList(engine);

    if (Settings::isMDJobType())
    {
        setupThermostat(engine);

        setupManostat(engine);
    }

    if (Settings::isMMActivated())
    {
        setupPotential(engine);

        setupIntraNonBonded(engine);

        setupForceField(engine);
    }

    setupConstraints(engine);

    if (Settings::isMDJobType())
        setupRingPolymer(engine);

    if (Settings::isQMMMActivated())
        setupHybrid(engine);

    if (Settings::isOptJobType())
        setupOptimizer(engine);
}