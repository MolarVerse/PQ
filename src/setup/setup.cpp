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

#include "celllistSetup.hpp"        // for setupCellList
#include "constraintsSetup.hpp"     // for setupConstraints
#include "engine.hpp"               // for Engine
#include "forceFieldSettings.hpp"   // for ForceFieldSettings
#include "forceFieldSetup.hpp"      // for setupForceField
#include "globalTimer.hpp"
#include "guffDatReader.hpp"                // for readGuffDat, readInput
#include "hybridSetup.hpp"                  // for setupQMMM
#include "inputFileReader.hpp"              // for readInputFile
#include "intraNonBondedReader.hpp"         // for readIntraNonBondedFile
#include "intraNonBondedSetup.hpp"          // for setupIntraNonBonded
#include "manostatSetup.hpp"                // for setupManostat
#include "moldescriptorReader.hpp"          // for readMolDescriptor
#include "optimizerSetup.hpp"               // for setupOptimizer
#include "outputFilesSetup.hpp"             // for setupOutputFiles
#include "parameterFileReader.hpp"          // for readParameterFile
#include "potentialSetup.hpp"               // for setupPotential
#include "qmSetup.hpp"                      // for setupQM
#include "randomNumberGeneratorSetup.hpp"   // for setupRandomNumberGenerator
#include "resetKineticsSetup.hpp"           // for setupResetKinetics
#include "restartFileReader.hpp"            // for readRestartFile
#include "ringPolymerSetup.hpp"             // for setupRingPolymer
#include "settings.hpp"                     // for Settings
#include "simulationBoxSetup.hpp"           // for setupSimulationBox
#include "thermostatSetup.hpp"              // for setupThermostat
#include "topologyReader.hpp"               // for readTopologyFile
#include "velocityVerlet.hpp"
#include "waterModelSettings.hpp"   // for WaterModelSettings
#include "waterModelSetup.hpp"      // for setupWaterModel

namespace setup
{

    /**
     * @brief setup the engine
     *
     * @param inputFileName
     * @param engine
     */
    void setupRequestedJob(
        const std::string& inputFileName,
        engine::Engine&    engine
    )
    {
        auto _ = scopedTimer(TimerId::Setup, "TotalSetup");

        startSetup();

        input::readInputFile(inputFileName, engine);

        setupOutputFiles(engine);

        readFiles(engine);

        setupEngine(engine);

        // needs setup of engine before reading guff.dat
        input::guffdat::readGuffDat(engine);

        endSetup(engine);
    }

    /**
     * @brief start the setup
     *
     */
    void startSetup() { out::StdoutOutput::writeHeader(); }

    /**
     * @brief end the setup
     *
     * @param engine
     */
    void endSetup(engine::Engine& engine)
    {
        out::StdoutOutput::writeSetupCompleted();
        engine.getLogOutput().writeSetupCompleted();
    }

    /**
     * @brief reads all the files needed for the simulation
     *
     * @param engine
     */
    void readFiles(engine::Engine& engine)
    {
        input::molDescriptor::readMolDescriptor(engine);

        input::restartFile::readRestartFile(engine);

        input::topology::readTopologyFile(engine);

        input::parameterFile::readParameterFile(engine);

        input::intraNonBondedReader::readIntraNonBondedFile(engine);
    }

    /**
     * @brief setup the engine
     *
     * @param engine
     */
    void setupEngine(engine::Engine& engine)
    {
        if (settings::Settings::isQMActivated())
            setupQM(engine);

        if (settings::Settings::isMDJobType())
        {
            switch (settings::Settings::getIntegratorType())
            {
                case settings::IntegratorType::VELOCITY_VERLET:
                {
                    auto& mdEngine = dynamic_cast<engine::MDEngine&>(engine);
                    mdEngine.makeIntegrator(integrator::VelocityVerlet());
                    break;
                }
                case settings::IntegratorType::NONE:
                {
                    throw exc::InputFileException(
                        "Integrator is not set for MD simulation - please set "
                        "it "
                        "in the input file"
                    );
                }
            }
            setupRandomNumberGenerator(engine);
            setupResetKinetics(engine);
        }

        setupSimulationBox(engine);

        setupCellList(engine);

        if (settings::Settings::isMDJobType())
        {
            setupThermostat(engine);

            setupManostat(engine);
        }

        if (settings::Settings::isMMActivated())
        {
            setupPotential(engine);

            setupIntraNonBonded(engine);
        }

        if (settings::ForceFieldSettings::isActive())
            setupForceField(engine);

        if (settings::WaterModelSettings::isWaterModelSet())
            setupWaterModel(engine);

        setupConstraints(engine);

        if (settings::Settings::isMDJobType())
            setupRingPolymer(engine);

        if (settings::Settings::isHybridJobtype())
            setupHybrid(engine);

        if (settings::Settings::isOptJobType())
            setupOptimizer(engine);

        engine.getLogOutput().flushQueuedWarnings();
    }

}   // namespace setup
