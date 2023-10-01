/*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
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
using namespace input;

/**
 * @brief setup the engine
 *
 * @param inputFileName
 * @param engine
 */
void setup::setupSimulation(const std::string &inputFileName, Engine &engine)
{
    readInputFile(inputFileName, engine);

    setupOutputFiles(engine);

    readFiles(engine);

    setupEngine(engine);

    // needs setup of engine before reading guff.dat
    guffdat::readGuffDat(engine);

    engine.getStdoutOutput().writeSetup("FULL simulation complete!");
    engine.getLogOutput().writeSetup("FULL simulation complete!");
}

/**
 * @brief reads all the files needed for the simulation
 *
 * @param inputFileName
 * @param engine
 */
void setup::readFiles(Engine &engine)
{
    molDescriptor::readMolDescriptor(engine);

    restartFile::readRestartFile(engine);

    topology::readTopologyFile(engine);

    parameterFile::readParameterFile(engine);

    input::intraNonBonded::readIntraNonBondedFile(engine);
}

/**
 * @brief setup the engine
 *
 * @param engine
 */
void setup::setupEngine(Engine &engine)
{
    if (settings::Settings::isQMActivated())
        setupQM(dynamic_cast<engine::QMMDEngine &>(engine));

    resetKinetics::setupResetKinetics(engine);

    simulationBox::setupSimulationBox(engine);

    setupCellList(engine);

    setupThermostat(engine);

    setupManostat(engine);

    if (settings::Settings::isMMActivated())
    {
        setupPotential(engine);   // has to be after simulationBox setup due to coulomb radius cutoff

        setupIntraNonBonded(engine);

        setupForceField(engine);
    }

    setupConstraints(engine);

    if (settings::Settings::isRingPolymerMDActivated())
        setupRingPolymer(dynamic_cast<engine::RingPolymerEngine &>(engine));
}