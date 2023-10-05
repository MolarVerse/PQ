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

#include "outputFilesSetup.hpp"

#include "energyOutput.hpp"                   // for EnergyOutput
#include "engine.hpp"                         // for Engine
#include "infoOutput.hpp"                     // for InfoOutput
#include "logOutput.hpp"                      // for LogOutput
#include "momentumOutput.hpp"                 // for MomentumOutput
#include "outputFileSettings.hpp"             // for OutputFileSettings
#include "ringPolymerRestartFileOutput.hpp"   // for RingPolymerRestartFileOutput
#include "ringPolymerTrajectoryOutput.hpp"    // for RingPolymerTrajectoryOutput
#include "rstFileOutput.hpp"                  // for RstFileOutput
#include "settings.hpp"                       // for Settings
#include "stdoutOutput.hpp"                   // for StdoutOutput
#include "trajectoryOutput.hpp"               // for TrajectoryOutput

#include <string>   // for string

using setup::OutputFilesSetup;

/**
 * @brief wrapper function to setup output files
 *
 */
void setup::setupOutputFiles(engine::Engine &engine)
{
    engine.getStdoutOutput().writeSetup("output files");

    OutputFilesSetup outputFilesSetup(engine);
    outputFilesSetup.setup();
}

/**
 * @brief setup output files
 *
 */
void OutputFilesSetup::setup()
{
    if (settings::OutputFileSettings::isFilePrefixSet())
        settings::OutputFileSettings::replaceDefaultValues(settings::OutputFileSettings::getFilePrefix());
    else
    {
        const auto prefix = settings::OutputFileSettings::determineMostCommonPrefix();
        settings::OutputFileSettings::replaceDefaultValues(prefix);
    }

    _engine.getRstFileOutput().setFilename(settings::OutputFileSettings::getRestartFileName());
    _engine.getEnergyOutput().setFilename(settings::OutputFileSettings::getEnergyFileName());
    _engine.getXyzOutput().setFilename(settings::OutputFileSettings::getTrajectoryFileName());
    _engine.getLogOutput().setFilename(settings::OutputFileSettings::getLogFileName());
    _engine.getInfoOutput().setFilename(settings::OutputFileSettings::getInfoFileName());
    _engine.getVelOutput().setFilename(settings::OutputFileSettings::getVelocityFileName());
    _engine.getForceOutput().setFilename(settings::OutputFileSettings::getForceFileName());
    _engine.getChargeOutput().setFilename(settings::OutputFileSettings::getChargeFileName());
    _engine.getMomentumOutput().setFilename(settings::OutputFileSettings::getMomentumFileName());

    if (settings::Settings::isRingPolymerMDActivated())
    {
        _engine.getRingPolymerRstFileOutput().setFilename(settings::OutputFileSettings::getRingPolymerRestartFileName());
        _engine.getRingPolymerXyzOutput().setFilename(settings::OutputFileSettings::getRingPolymerTrajectoryFileName());
        _engine.getRingPolymerVelOutput().setFilename(settings::OutputFileSettings::getRingPolymerVelocityFileName());
        _engine.getRingPolymerForceOutput().setFilename(settings::OutputFileSettings::getRingPolymerForceFileName());
        _engine.getRingPolymerChargeOutput().setFilename(settings::OutputFileSettings::getRingPolymerChargeFileName());
    }

    _engine.getLogOutput().writeHeader();
}