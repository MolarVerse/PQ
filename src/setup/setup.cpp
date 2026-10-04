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

        input::readInputFile(inputFileName);

        // needs to happen before readFiles(): the parameter file reader
        // dynamic_casts the non-Coulomb potential to its concrete type
        // while parsing the NONCOULOMBICS section
        if (settings::GeneralSettings::isMMActivated())
            setupNonCoulombPotentialType(engine);

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
        if (settings::GeneralSettings::isQMActivated())
            setupQM(engine);

        if (settings::GeneralSettings::isMDJobType())
        {
            switch (settings::GeneralSettings::getIntegratorType())
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

        if (settings::GeneralSettings::isMDJobType())
        {
            setupThermostat(engine);

            setupManostat(engine);
        }

        if (settings::GeneralSettings::isMMActivated())
        {
            setupPotential(engine);

            setupIntraNonBonded(engine);
        }

        if (settings::ForceFieldSettings::isActive())
            setupForceField(engine);

        if (settings::WaterModelSettings::isWaterModelSet())
            setupWaterModel(engine);

        setupConstraints(engine);

        if (settings::GeneralSettings::isMDJobType())
            setupRingPolymer(engine);

        if (settings::GeneralSettings::isHybridJobtype())
            setupHybrid(engine);

        if (settings::GeneralSettings::isOptJobType())
            setupOptimizer(engine);

        engine.getLogOutput().flushQueuedWarnings();
    }

}   // namespace setup
