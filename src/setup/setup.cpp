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

#include <format>

#include "celllistSetup.hpp"
#include "constraintsSetup.hpp"
#include "engine.hpp"
#include "forceFieldSettings.hpp"
#include "forceFieldSetup.hpp"
#include "generalSettings.hpp"
#include "globalTimer.hpp"
#include "guffDatReader.hpp"
#include "hybridSetup.hpp"
#include "inputFileReader.hpp"
#include "intraNonBondedReader.hpp"
#include "intraNonBondedSetup.hpp"
#include "logOutput.hpp"
#include "manostatSetup.hpp"
#include "moldescriptorReader.hpp"
#include "optimizerSetup.hpp"
#include "outputFilesSetup.hpp"
#include "parameterFileReader.hpp"
#include "potentialSetup.hpp"
#include "qmSetup.hpp"
#include "randomNumberGeneratorSetup.hpp"
#include "resetKineticsSetup.hpp"
#include "restartFileReader.hpp"
#include "ringPolymerSetup.hpp"
#include "simulationBoxSetup.hpp"
#include "stdoutOutput.hpp"
#include "thermostatSetup.hpp"
#include "topologyReader.hpp"
#include "velocityVerlet.hpp"
#include "waterModelSettings.hpp"
#include "waterModelSetup.hpp"

namespace setup
{

    /**
     * @brief setup the engine
     *
     * @param engine The engine to set up.
     * @param settings Reference to the Settings object containing all necessary
     * simulation settings.
     */
    void setupRequestedJob(engine::Engine& engine, Settings& settings)
    {
        auto _ = scopedTimer(TimerId::Setup, "TotalSetup");

        startSetup();

        setupOutputFiles(engine);

        writeEngineSetupInfo(
            engine.getLogOutput(),
            engine.getSettings().generalSettings.getJobtype()
        );

        // needs to happen before readFiles(): the parameter file reader
        // dynamic_casts the non-Coulomb potential to its concrete type
        // while parsing the NONCOULOMBICS section
        if (isMMActivated(engine.getSettings().generalSettings.getJobtype()))
            setupNonCoulombPotentialType(engine);

        readFiles(engine, settings);

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
     * @param settings Reference to the Settings object containing all necessary
     * simulation settings.
     */
    void readFiles(engine::Engine& engine, Settings& settings)
    {
        input::molDescriptor::readMolDescriptor(engine);

        input::restartFile::readRestartFile(
            engine,
            settings.noseHooverSettings,
            settings.timings,
            settings.generalSettings
        );

        input::topology::readTopologyFile(engine);

        input::parameterFile::readParameterFile(
            engine,
            settings.potentialSettings
        );

        input::intraNonBondedReader::readIntraNonBondedFile(engine);
    }

    /**
     * @brief setup the engine
     *
     * @param engine
     */
    void setupEngine(engine::Engine& engine)
    {
        const auto& settings = engine.getSettings();

        if (isQMActivated(engine.getSettings().generalSettings.getJobtype()))
            setupQM(engine);

        if (isMDJobType(engine.getSettings().generalSettings.getJobtype()))
        {
            switch (GeneralSettings::getIntegratorType())
            {
                case IntegratorType::VELOCITY_VERLET:
                {
                    auto& mdEngine = dynamic_cast<engine::MDEngine&>(engine);
                    mdEngine.makeIntegrator(integrator::VelocityVerlet());
                    break;
                }
                case IntegratorType::NONE:
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

        if (isMDJobType(engine.getSettings().generalSettings.getJobtype()))
        {
            setupThermostat(engine);

            setupManostat(engine);
        }

        if (isMMActivated(engine.getSettings().generalSettings.getJobtype()))
        {
            setupPotential(engine);

            setupIntraNonBonded(engine);
        }

        if (settings.forceFieldSettings.isActive())
            setupForceField(engine);

        if (settings.waterModelSettings.isWaterModelSetAndNotNone())
            setupWaterModel(engine);

        setupConstraints(engine);

        if (isMDJobType(engine.getSettings().generalSettings.getJobtype()))
            setupRingPolymer(engine);

        if (isHybridJobtype(engine.getSettings().generalSettings.getJobtype()))
            setupHybrid(engine);

        if (isOptJobType(engine.getSettings().generalSettings.getJobtype()))
            setupOptimizer(engine);

        engine.getLogOutput().flushQueuedWarnings();
    }

    /**
     * @brief Writes the engine setup information to the log output.
     *
     * This function logs the general engine setup information, including
     * the time step, starting step count, and the total number of steps.
     *
     * @param logOutput The log output object to which the engine setup
     * information will be written.
     * @param jobType The type of job for which the engine is being set up.
     */
    void writeEngineSetupInfo(out::LogOutput& logOutput, JobType jobType)
    {
        const auto msg = std::format(
            "Engine for jobtype: {}",
            JobTypeMeta::toString(jobType)
        );
        logOutput.writeSetup(msg);
        out::StdoutOutput::writeSetup(msg);
    }

}   // namespace setup
