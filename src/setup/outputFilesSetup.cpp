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
#include "outputFileSettings.hpp"             // for OutputFileSettings
#include "ringPolymerRestartFileOutput.hpp"   // for RingPolymerRestartFileOutput
#include "ringPolymerTrajectoryOutput.hpp"    // for RingPolymerTrajectoryOutput
#include "rstFileOutput.hpp"                  // for RstFileOutput
#include "settings.hpp"                       // for Settings
#include "stdoutOutput.hpp"                   // for StdoutOutput
#include "trajectoryOutput.hpp"               // for TrajectoryOutput

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

    engine.getLogOutput().writeHeader();
}

/**
 * @brief setup output files
 *
 */
void OutputFilesSetup::setup()
{
    if (settings::OutputFileSettings::isFilePrefixSet())
        settings::OutputFileSettings::replaceDefaultValues(
            settings::OutputFileSettings::getFilePrefix()
        );
    else
    {
        const auto prefix =
            settings::OutputFileSettings::determineMostCommonPrefix();
        settings::OutputFileSettings::replaceDefaultValues(prefix);
    }

    _engine.getLogOutput().setFilename(
        settings::OutputFileSettings::getLogFileName()
    );

    _engine.getTimingsOutput().setFilename(
        settings::OutputFileSettings::getTimingsFileName()
    );

    if (settings::Settings::isMDJobType())
    {
        dynamic_cast<engine::MDEngine &>(_engine)
            .getRstFileOutput()
            .setFilename(settings::OutputFileSettings::getRestartFileName());
        dynamic_cast<engine::MDEngine &>(_engine).getEnergyOutput().setFilename(
            settings::OutputFileSettings::getEnergyFileName()
        );
        dynamic_cast<engine::MDEngine &>(_engine)
            .getInstantEnergyOutput()
            .setFilename(settings::OutputFileSettings::getInstantEnergyFileName(
            ));
        dynamic_cast<engine::MDEngine &>(_engine).getXyzOutput().setFilename(
            settings::OutputFileSettings::getTrajectoryFileName()
        );
        dynamic_cast<engine::MDEngine &>(_engine).getInfoOutput().setFilename(
            settings::OutputFileSettings::getInfoFileName()
        );
        dynamic_cast<engine::MDEngine &>(_engine).getVelOutput().setFilename(
            settings::OutputFileSettings::getVelocityFileName()
        );
        dynamic_cast<engine::MDEngine &>(_engine).getForceOutput().setFilename(
            settings::OutputFileSettings::getForceFileName()
        );
        dynamic_cast<engine::MDEngine &>(_engine).getChargeOutput().setFilename(
            settings::OutputFileSettings::getChargeFileName()
        );
        dynamic_cast<engine::MDEngine &>(_engine)
            .getMomentumOutput()
            .setFilename(settings::OutputFileSettings::getMomentumFileName());
        dynamic_cast<engine::MDEngine &>(_engine).getVirialOutput().setFilename(
            settings::OutputFileSettings::getVirialFileName()
        );
        dynamic_cast<engine::MDEngine &>(_engine).getStressOutput().setFilename(
            settings::OutputFileSettings::getStressFileName()
        );
        dynamic_cast<engine::MDEngine &>(_engine)
            .getBoxFileOutput()
            .setFilename(settings::OutputFileSettings::getBoxFileName());

        if (settings::Settings::isRingPolymerMDActivated())
        {
            dynamic_cast<engine::MDEngine &>(_engine)
                .getRingPolymerRstFileOutput()
                .setFilename(
                    settings::OutputFileSettings::getRingPolymerRestartFileName(
                    )
                );
            dynamic_cast<engine::MDEngine &>(_engine)
                .getRingPolymerXyzOutput()
                .setFilename(settings::OutputFileSettings::
                                 getRingPolymerTrajectoryFileName());
            dynamic_cast<engine::MDEngine &>(_engine)
                .getRingPolymerVelOutput()
                .setFilename(settings::OutputFileSettings::
                                 getRingPolymerVelocityFileName());
            dynamic_cast<engine::MDEngine &>(_engine)
                .getRingPolymerForceOutput()
                .setFilename(
                    settings::OutputFileSettings::getRingPolymerForceFileName()
                );
            dynamic_cast<engine::MDEngine &>(_engine)
                .getRingPolymerChargeOutput()
                .setFilename(
                    settings::OutputFileSettings::getRingPolymerChargeFileName()
                );
            dynamic_cast<engine::MDEngine &>(_engine)
                .getRingPolymerEnergyOutput()
                .setFilename(
                    settings::OutputFileSettings::getRingPolymerEnergyFileName()
                );
        }
    }
}