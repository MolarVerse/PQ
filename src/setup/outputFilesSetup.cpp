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

#include "engine.hpp"               // for Engine
#include "hessianEngine.hpp"        // for HessianEngine
#include "hessianSettings.hpp"      // for HessianSettings
#include "infoOutput.hpp"           // for InfoOutput
#include "logOutput.hpp"            // for LogOutput
#include "mdEngine.hpp"             // for MDEngine
#include "optEngine.hpp"            // for OptEngine
#include "outputFileSettings.hpp"   // for OutputFileSettings
#include "settings.hpp"             // for Settings
#include "stdoutOutput.hpp"         // for StdoutOutput
#include "timingsSettings.hpp"      // for TimingsSettings
#include "trajectoryOutput.hpp"     // for TrajectoryOutput

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
        const auto isPrefixSet =
            settings::OutputFileSettings::isFilePrefixSet();
        auto prefix = std::string();

        if (isPrefixSet)
            prefix = settings::OutputFileSettings::getFilePrefix();
        else
            prefix = settings::OutputFileSettings::determineMostCommonPrefix();

        settings::OutputFileSettings::replaceDefaultValues(prefix);

        const auto logFileName = settings::OutputFileSettings::getLogFileName();
        const auto timingsFileName =
            settings::OutputFileSettings::getTimingsFileName();
        const auto restartFileName =
            settings::OutputFileSettings::getRestartFileName();
        const auto energyFileName =
            settings::OutputFileSettings::getEnergyFileName();
        const auto xyzFileName =
            settings::OutputFileSettings::getTrajectoryFileName();
        const auto infoFileName =
            settings::OutputFileSettings::getInfoFileName();
        const auto forceFileName =
            settings::OutputFileSettings::getForceFileName();

        _engine.getLogOutput().setFilename(logFileName);
        _engine.getTimingsOutput().setFilename(timingsFileName);
        _engine.getRstFileOutput().setFilename(restartFileName);
        _engine.getEnergyOutput().setFilename(energyFileName);
        _engine.getXyzOutput().setFilename(xyzFileName);
        _engine.getInfoOutput().setFilename(infoFileName);
        _engine.getForceOutput().setFilename(forceFileName);

        if (settings::Settings::isMDJobType())
        {
            auto &mdEngine = dynamic_cast<engine::MDEngine &>(_engine);

            const auto instEnFile =
                settings::OutputFileSettings::getInstantEnergyFileName();
            const auto velFile =
                settings::OutputFileSettings::getVelocityFileName();
            const auto chargeFile =
                settings::OutputFileSettings::getChargeFileName();
            const auto momFile =
                settings::OutputFileSettings::getMomentumFileName();
            const auto virialFile =
                settings::OutputFileSettings::getVirialFileName();
            const auto stressFile =
                settings::OutputFileSettings::getStressFileName();
            const auto boxFile = settings::OutputFileSettings::getBoxFileName();

            mdEngine.getInstantEnergyOutput().setFilename(instEnFile);
            mdEngine.getVelOutput().setFilename(velFile);
            mdEngine.getChargeOutput().setFilename(chargeFile);
            mdEngine.getMomentumOutput().setFilename(momFile);
            mdEngine.getVirialOutput().setFilename(virialFile);
            mdEngine.getStressOutput().setFilename(stressFile);
            mdEngine.getBoxFileOutput().setFilename(boxFile);

            if (settings::Settings::isHybridJobtype())
            {
                const auto hybridCenterFile =
                    settings::OutputFileSettings::getHybridCenterFileName();
                mdEngine.getXyzHybridCenterOutput().setFilename(
                    hybridCenterFile
                );
            }

            if (settings::OutputFileSettings::getIncludeOutputMetadata())
            {
                const auto timeStep = settings::TimingsSettings::getTimeStep();
                _engine.getEnergyOutput().writeHeader(timeStep);
                mdEngine.getInstantEnergyOutput().writeHeader(timeStep);
            }

            if (settings::Settings::isRingPolymerMDActivated())
            {
                const auto rstFile_ =
                    settings::OutputFileSettings::getRPMDRestartFileName();
                const auto xyzFile_ =
                    settings::OutputFileSettings::getRPMDTrajFileName();
                const auto velFile_ =
                    settings::OutputFileSettings::getRPMDVelocityFileName();
                const auto forceFile_ =
                    settings::OutputFileSettings::getRPMDForceFileName();
                const auto chargeFile_ =
                    settings::OutputFileSettings::getRPMDChargeFileName();
                const auto energyFile_ =
                    settings::OutputFileSettings::getRPMDEnergyFileName();

                mdEngine.getRingPolymerRstFileOutput().setFilename(rstFile_);
                mdEngine.getRingPolymerXyzOutput().setFilename(xyzFile_);
                mdEngine.getRingPolymerVelOutput().setFilename(velFile_);
                mdEngine.getRingPolymerForceOutput().setFilename(forceFile_);
                mdEngine.getRingPolymerChargeOutput().setFilename(chargeFile_);
                mdEngine.getRingPolymerEnergyOutput().setFilename(energyFile_);
            }
        }

        if (settings::Settings::isOptJobType())
        {
            auto &optEngine = dynamic_cast<engine::OptEngine &>(_engine);

            const auto optFileName =
                settings::OutputFileSettings::getOptFileName();

            optEngine.getOptOutput().setFilename(optFileName);
        }

        if (settings::Settings::getJobtype() == JobType::MM_HESSIAN &&
            settings::HessianSettings::optimizeBeforeHessian())
        {
            auto &hessianEngine =
                dynamic_cast<engine::HessianEngine &>(_engine);

            const auto optFileName =
                settings::OutputFileSettings::getOptFileName();

            hessianEngine.getOptOutput().setFilename(optFileName);
        }
    }

}   // namespace setup
