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

#include "outputFilesSetup.hpp"

#include <string>   // for string

#include "boxOutput.hpp"                      // for BoxFileOutput
#include "energyOutput.hpp"                   // for EnergyOutput
#include "engine.hpp"                         // for Engine
#include "infoOutput.hpp"                     // for InfoOutput
#include "logOutput.hpp"                      // for LogOutput
#include "mdEngine.hpp"                       // for MDEngine
#include "momentumOutput.hpp"                 // for MomentumOutput
#include "optEngine.hpp"                      // for OptEngine
#include "outputFileSettings.hpp"             // for OutputFileSettings
#include "ringPolymerRestartFileOutput.hpp"   // for RingPolymerRestartFileOutput
#include "ringPolymerTrajectoryOutput.hpp"    // for RingPolymerTrajectoryOutput
#include "rstFileOutput.hpp"                  // for RstFileOutput
#include "settings.hpp"                       // for Settings
#include "stdoutOutput.hpp"                   // for StdoutOutput
#include "trajectoryOutput.hpp"               // for TrajectoryOutput

using setup::OutputFilesSetup;
using namespace settings;
using namespace engine;

/**
 * @brief wrapper function to setup output files
 *
 */
void setup::setupOutputFiles(Engine &engine)
{
    engine.getStdoutOutput().writeSetup("Output Files");

    OutputFilesSetup outputFilesSetup(engine);
    outputFilesSetup.setup();

    engine.getLogOutput().writeHeader();
}

/**
 * @brief Construct a new Output Files Setup object
 *
 * @param engine
 */
OutputFilesSetup::OutputFilesSetup(Engine &engine) : _engine(engine){};

/**
 * @brief setup output files
 *
 */
void OutputFilesSetup::setup()
{
    const auto isPrefixSet = OutputFileSettings::isFilePrefixSet();
    auto       prefix      = std::string();

    if (isPrefixSet)
        prefix = OutputFileSettings::getFilePrefix();
    else
        prefix = OutputFileSettings::determineMostCommonPrefix();

    OutputFileSettings::replaceDefaultValues(prefix);

    const auto logFileName     = OutputFileSettings::getLogFileName();
    const auto timingsFileName = OutputFileSettings::getTimingsFileName();
    const auto restartFileName = OutputFileSettings::getRestartFileName();
    const auto energyFileName  = OutputFileSettings::getEnergyFileName();
    const auto xyzFileName     = OutputFileSettings::getTrajectoryFileName();
    const auto infoFileName    = OutputFileSettings::getInfoFileName();
    const auto forceFileName   = OutputFileSettings::getForceFileName();

    _engine.getLogOutput().setFilename(logFileName);
    _engine.getTimingsOutput().setFilename(timingsFileName);
    _engine.getRstFileOutput().setFilename(restartFileName);
    _engine.getEnergyOutput().setFilename(energyFileName);
    _engine.getXyzOutput().setFilename(xyzFileName);
    _engine.getInfoOutput().setFilename(infoFileName);
    _engine.getForceOutput().setFilename(forceFileName);

    if (Settings::isMDJobType())
    {
        auto &mdEngine = dynamic_cast<MDEngine &>(_engine);

        const auto instEnFile = OutputFileSettings::getInstantEnergyFileName();
        const auto velFile    = OutputFileSettings::getVelocityFileName();
        const auto chargeFile = OutputFileSettings::getChargeFileName();
        const auto momFile    = OutputFileSettings::getMomentumFileName();
        const auto virialFile = OutputFileSettings::getVirialFileName();
        const auto stressFile = OutputFileSettings::getStressFileName();
        const auto boxFile    = OutputFileSettings::getBoxFileName();

        mdEngine.getInstantEnergyOutput().setFilename(instEnFile);
        mdEngine.getVelOutput().setFilename(velFile);
        mdEngine.getChargeOutput().setFilename(chargeFile);
        mdEngine.getMomentumOutput().setFilename(momFile);
        mdEngine.getVirialOutput().setFilename(virialFile);
        mdEngine.getStressOutput().setFilename(stressFile);
        mdEngine.getBoxFileOutput().setFilename(boxFile);

        if (Settings::isRingPolymerMDActivated())
        {
            const auto RstFile = OutputFileSettings::getRPMDRestartFileName();
            const auto xyzFile = OutputFileSettings::getRPMDTrajFileName();
            const auto velFile = OutputFileSettings::getRPMDVelocityFileName();
            const auto forceFile  = OutputFileSettings::getRPMDForceFileName();
            const auto chargeFile = OutputFileSettings::getRPMDChargeFileName();
            const auto energyFile = OutputFileSettings::getRPMDEnergyFileName();

            mdEngine.getRingPolymerRstFileOutput().setFilename(RstFile);
            mdEngine.getRingPolymerXyzOutput().setFilename(xyzFile);
            mdEngine.getRingPolymerVelOutput().setFilename(velFile);
            mdEngine.getRingPolymerForceOutput().setFilename(forceFile);
            mdEngine.getRingPolymerChargeOutput().setFilename(chargeFile);
            mdEngine.getRingPolymerEnergyOutput().setFilename(energyFile);
        }
    }

    if (Settings::isOptJobType())
    {
        auto &optEngine = dynamic_cast<OptEngine &>(_engine);

        const auto optFileName = OutputFileSettings::getOptFileName();

        optEngine.getOptOutput().setFilename(optFileName);
    }
}