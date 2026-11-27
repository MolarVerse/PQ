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
#include "units.hpp"

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
            outputFileSettings.get_overwriteOutputFiles();

        const auto logFileName     = outputFileSettings.get_logFile();
        const auto timingsFileName = outputFileSettings.get_timeFile();
        const auto restartFileName = outputFileSettings.get_rstFile();
        const auto energyFileName  = outputFileSettings.get_energyFile();
        const auto xyzFileName     = outputFileSettings.get_trajFile();
        const auto infoFileName    = outputFileSettings.get_infoFile();
        const auto forceFileName   = outputFileSettings.get_forceFile();

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
        _engine.getInfoOutput().setFilename(infoFileName, overwriteOutputFiles);

        // extxyz, h5md or ase replace the xyz, velocity, force and charge files
        // of MD runs
        const auto format = outputFileSettings.get_trajFormat();
        const auto textTrajectory =
            !isMDJobType(_engine.getSettings().generalSettings.getJobtype()) ||
            format == TrajectoryFormat::XYZ;

        if (textTrajectory)
        {
            _engine.getXyzOutput().setFilename(
                xyzFileName,
                overwriteOutputFiles
            );
            _engine.getForceOutput().setFilename(
                forceFileName,
                overwriteOutputFiles
            );
        }

        if (isMDJobType(_engine.getSettings().generalSettings.getJobtype()))
        {
            auto &mdEngine = dynamic_cast<engine::MDEngine &>(_engine);

            const auto instEnFile = outputFileSettings.get_instEnFile();
            const auto velFile    = outputFileSettings.get_velFile();
            const auto chargeFile = outputFileSettings.get_chargeFile();
            const auto momFile    = outputFileSettings.get_momFile();
            const auto virialFile = outputFileSettings.get_virialFile();
            const auto stressFile = outputFileSettings.get_stressFile();
            const auto boxFile    = outputFileSettings.get_boxFile();

            mdEngine.getInstantEnergyOutput().setFilename(
                instEnFile,
                overwriteOutputFiles
            );
            if (format == TrajectoryFormat::EXTXYZ)
            {
                mdEngine.getEngineOutput().openExtxyzOutput(
                    outputFileSettings.get_extxyzFile(),
                    overwriteOutputFiles
                );
            }
            else if (format == TrajectoryFormat::H5MD)
            {
                const auto &timeStep =
                    _engine.getSettings().timings.getTimeStep();

                mdEngine.getEngineOutput().openH5mdOutput(
                    outputFileSettings.get_h5mdFile(),
                    overwriteOutputFiles,
                    timeStep.numerical_value_in(fs)
                );
            }
            else if (format == TrajectoryFormat::ASE)
            {
                mdEngine.getEngineOutput().openAseTrajOutput(
                    outputFileSettings.get_aseTrajFile(),
                    overwriteOutputFiles
                );
            }
            else
            {
                mdEngine.getVelOutput().setFilename(
                    velFile,
                    overwriteOutputFiles
                );
                mdEngine.getChargeOutput().setFilename(
                    chargeFile,
                    overwriteOutputFiles
                );
            }

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

            if (isHybridJobtype(
                    _engine.getSettings().generalSettings.getJobtype()
                ))
            {
                const auto hybridCenterFile =
                    outputFileSettings.get_hybridCenterFile();
                mdEngine.getXyzHybridCenterOutput().setFilename(
                    hybridCenterFile,
                    overwriteOutputFiles
                );
            }

            if (outputFileSettings.get_includeOutputMetadata())
            {
                const auto timeStep =
                    _engine.getSettings().timings.getTimeStep();
                _engine.getEnergyOutput().writeHeader(timeStep);
                mdEngine.getInstantEnergyOutput().writeHeader(timeStep);
            }

            if (isRingPolymerMDActivated(
                    _engine.getSettings().generalSettings.getJobtype()
                ))
            {
                const auto rstFile_   = outputFileSettings.get_rpmdRstFile();
                const auto xyzFile_   = outputFileSettings.get_rpmdTrajFile();
                const auto velFile_   = outputFileSettings.get_rpmdVelFile();
                const auto forceFile_ = outputFileSettings.get_rpmdForceFile();
                const auto chargeFile_ =
                    outputFileSettings.get_rpmdChargeFile();
                const auto energyFile_ =
                    outputFileSettings.get_rpmdEnergyFile();
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

        if (isOptJobType(_engine.getSettings().generalSettings.getJobtype()))
        {
            auto &optEngine = dynamic_cast<engine::OptEngine &>(_engine);

            const auto optFileName = outputFileSettings.get_optFile();

            optEngine.getOptOutput().setFilename(
                optFileName,
                overwriteOutputFiles
            );
        }

        if (_engine.getSettings().generalSettings.getJobtype() ==
                JobType::MM_HESSIAN &&
            _engine.getSettings().hessianSettings.get_optimizeBeforeHessian())
        {
            auto &hessianEngine =
                dynamic_cast<engine::HessianEngine &>(_engine);

            const auto optFileName = outputFileSettings.get_optFile();

            hessianEngine.getOptOutput().setFilename(
                optFileName,
                overwriteOutputFiles
            );
        }
    }

}   // namespace setup
