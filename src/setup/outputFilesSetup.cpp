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
        const auto xyzFileName   = OutputFileSettings::getTrajectoryFileName();
        const auto infoFileName  = OutputFileSettings::getInfoFileName();
        const auto forceFileName = OutputFileSettings::getForceFileName();

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
                OutputFileSettings::getInstantEnergyFileName();
            const auto velFile    = OutputFileSettings::getVelocityFileName();
            const auto chargeFile = OutputFileSettings::getChargeFileName();
            const auto momFile    = OutputFileSettings::getMomentumFileName();
            const auto virialFile = OutputFileSettings::getVirialFileName();
            const auto stressFile = OutputFileSettings::getStressFileName();
            const auto boxFile    = OutputFileSettings::getBoxFileName();

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
                    OutputFileSettings::getHybridCenterFileName();
                mdEngine.getXyzHybridCenterOutput().setFilename(
                    hybridCenterFile,
                    overwriteOutputFiles
                );
            }

            if (OutputFileSettings::getIncludeOutputMetadata())
            {
                const auto timeStep = settings::TimingsSettings::getTimeStep();
                _engine.getEnergyOutput().writeHeader(timeStep);
                mdEngine.getInstantEnergyOutput().writeHeader(timeStep);
            }

            if (settings::GeneralSettings::isRingPolymerMDActivated())
            {
                const auto rstFile_ =
                    OutputFileSettings::getRPMDRestartFileName();
                const auto xyzFile_ = OutputFileSettings::getRPMDTrajFileName();
                const auto velFile_ =
                    OutputFileSettings::getRPMDVelocityFileName();
                const auto forceFile_ =
                    OutputFileSettings::getRPMDForceFileName();
                const auto chargeFile_ =
                    OutputFileSettings::getRPMDChargeFileName();
                const auto energyFile_ =
                    OutputFileSettings::getRPMDEnergyFileName();

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

            const auto optFileName = OutputFileSettings::getOptFileName();

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

            const auto optFileName = OutputFileSettings::getOptFileName();

            hessianEngine.getOptOutput().setFilename(
                optFileName,
                overwriteOutputFiles
            );
        }
    }

}   // namespace setup
