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

#include <string>

#include "engine.hpp"
#include "generalSettings.hpp"
#include "hessianEngine.hpp"
#include "hessianSettings.hpp"
#include "infoOutput.hpp"
#include "logOutput.hpp"
#include "mdEngine.hpp"
#include "optEngine.hpp"
#include "outputFileSettings.hpp"
#include "stdoutOutput.hpp"
#include "timingsSettings.hpp"
#include "trajectoryOutput.hpp"

namespace setup
{

    /**
     * @brief wrapper function to setup output files
     *
     */
    void setupOutputFiles(engine::Engine &engine)
    {
        out::StdoutOutput::writeSetup("Output Files");

        OutputFilesSetup outputFilesSetup(engine);
        outputFilesSetup.setup();

        engine.getLogOutput().writeHeader();
    }

    /**
     * @brief Construct a new Output Files Setup object
     *
     * @param engine
     */
    OutputFilesSetup::OutputFilesSetup(engine::Engine &engine) : _engine(engine)
    {
    }

    /**
     * @brief setup output files
     *
     */
    void OutputFilesSetup::setup()
    {
        const auto &outputFileSettings = _engine.getSettings().outputFiles;

        const auto overwriteOutputFiles =
            outputFileSettings.getOverwriteOutputFiles();

        const auto logFileName     = outputFileSettings.getLogFileName();
        const auto timingsFileName = outputFileSettings.getTimingsFileName();
        const auto restartFileName = outputFileSettings.getRestartFileName();
        const auto energyFileName  = outputFileSettings.getEnergyFileName();
        const auto xyzFileName     = outputFileSettings.getTrajectoryFileName();
        const auto infoFileName    = outputFileSettings.getInfoFileName();
        const auto forceFileName   = outputFileSettings.getForceFileName();

        _engine.getLogOutput().setFilename(logFileName, overwriteOutputFiles);
        _engine.getTimingsOutput().setFilename(
            timingsFileName,
            overwriteOutputFiles
        );
        _engine.getRstFileOutput().setFilename(
            restartFileName,
            overwriteOutputFiles
        );
        _engine.getEnergyOutput().setFilename(
            energyFileName,
            overwriteOutputFiles
        );
        _engine.getXyzOutput().setFilename(xyzFileName, overwriteOutputFiles);
        _engine.getInfoOutput().setFilename(infoFileName, overwriteOutputFiles);
        _engine.getForceOutput().setFilename(
            forceFileName,
            overwriteOutputFiles
        );

        if (settings::GeneralSettings::isMDJobType())
        {
            auto &mdEngine = dynamic_cast<engine::MDEngine &>(_engine);

            const auto instEnFile =
                outputFileSettings.getInstantEnergyFileName();
            const auto velFile    = outputFileSettings.getVelocityFileName();
            const auto chargeFile = outputFileSettings.getChargeFileName();
            const auto momFile    = outputFileSettings.getMomentumFileName();
            const auto virialFile = outputFileSettings.getVirialFileName();
            const auto stressFile = outputFileSettings.getStressFileName();
            const auto boxFile    = outputFileSettings.getBoxFileName();

            mdEngine.getInstantEnergyOutput().setFilename(
                instEnFile,
                overwriteOutputFiles
            );
            mdEngine.getVelOutput().setFilename(velFile, overwriteOutputFiles);
            mdEngine.getChargeOutput().setFilename(
                chargeFile,
                overwriteOutputFiles
            );
            mdEngine.getMomentumOutput().setFilename(
                momFile,
                overwriteOutputFiles
            );
            mdEngine.getVirialOutput().setFilename(
                virialFile,
                overwriteOutputFiles
            );
            mdEngine.getStressOutput().setFilename(
                stressFile,
                overwriteOutputFiles
            );
            mdEngine.getBoxFileOutput().setFilename(
                boxFile,
                overwriteOutputFiles
            );

            if (settings::GeneralSettings::isHybridJobtype())
            {
                const auto hybridCenterFile =
                    outputFileSettings.getHybridCenterFileName();
                mdEngine.getXyzHybridCenterOutput().setFilename(
                    hybridCenterFile,
                    overwriteOutputFiles
                );
            }

            if (outputFileSettings.getIncludeOutputMetadata())
            {
                const auto timeStep = settings::TimingsSettings::getTimeStep();
                _engine.getEnergyOutput().writeHeader(timeStep);
                mdEngine.getInstantEnergyOutput().writeHeader(timeStep);
            }

            if (settings::GeneralSettings::isRingPolymerMDActivated())
            {
                const auto rstFile_ =
                    outputFileSettings.getRPMDRestartFileName();
                const auto xyzFile_ = outputFileSettings.getRPMDTrajFileName();
                const auto velFile_ =
                    outputFileSettings.getRPMDVelocityFileName();
                const auto forceFile_ =
                    outputFileSettings.getRPMDForceFileName();
                const auto chargeFile_ =
                    outputFileSettings.getRPMDChargeFileName();
                const auto energyFile_ =
                    outputFileSettings.getRPMDEnergyFileName();

                mdEngine.getRingPolymerRstFileOutput().setFilename(
                    rstFile_,
                    overwriteOutputFiles
                );
                mdEngine.getRingPolymerXyzOutput().setFilename(
                    xyzFile_,
                    overwriteOutputFiles
                );
                mdEngine.getRingPolymerVelOutput().setFilename(
                    velFile_,
                    overwriteOutputFiles
                );
                mdEngine.getRingPolymerForceOutput().setFilename(
                    forceFile_,
                    overwriteOutputFiles
                );
                mdEngine.getRingPolymerChargeOutput().setFilename(
                    chargeFile_,
                    overwriteOutputFiles
                );
                mdEngine.getRingPolymerEnergyOutput().setFilename(
                    energyFile_,
                    overwriteOutputFiles
                );
            }
        }

        if (settings::GeneralSettings::isOptJobType())
        {
            auto &optEngine = dynamic_cast<engine::OptEngine &>(_engine);

            const auto optFileName = outputFileSettings.getOptFileName();

            optEngine.getOptOutput().setFilename(
                optFileName,
                overwriteOutputFiles
            );
        }

        if (settings::GeneralSettings::getJobtype() == JobType::MM_HESSIAN &&
            _engine.getSettings().hessianSettings.optimizeBeforeHessian())
        {
            auto &hessianEngine =
                dynamic_cast<engine::HessianEngine &>(_engine);

            const auto optFileName = outputFileSettings.getOptFileName();

            hessianEngine.getOptOutput().setFilename(
                optFileName,
                overwriteOutputFiles
            );
        }
    }

}   // namespace setup
