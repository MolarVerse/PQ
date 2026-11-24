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

#include "restartFileReader.hpp"

#include <format>
#include <fstream>
#include <string>

#include "atomSection.hpp"
#include "boxSection.hpp"
#include "engine.hpp"
#include "exceptions.hpp"
#include "fileSettings.hpp"
#include "generalSettings.hpp"
#include "noseHooverSection.hpp"
#include "stepCountSection.hpp"
#include "stringUtilities.hpp"
#include "units.hpp"

namespace input::restartFile
{

    /**
     * @brief Construct a new Rst File Reader:: Rst File Reader object
     *
     * @details The constructor initializes the sections of the .rst file and
     * pushes them into a vector. It also sets the filename and the engine
     * object and with the filename it opens the file in the ifstream object
     * _fp.
     *
     *
     * @param filename
     * @param engine
     * @param noseHooverSettings Reference to the NoseHooverSettings object
     * @param timingsSettings Reference to the TimingsSettings object
     */
    RestartFileReader::RestartFileReader(
        const std::string  &filename,
        engine::Engine     &engine,
        NoseHooverSettings &noseHooverSettings,
        TimingsSettings    &timingsSettings
    )
        : _fileName(filename),
          _fp(filename),
          _engine(engine),
          _atomSection(std::make_unique<AtomSection>()),
          _timingsSettings(timingsSettings)
    {
        _sections.push_back(std::make_unique<BoxSection>());
        _sections.push_back(
            std::make_unique<NoseHooverSection>(noseHooverSettings)
        );
        _sections.push_back(
            std::make_unique<StepCountSection>(timingsSettings)
        );
    }

    /**
     * @brief Determines which section of the .rst file the line belongs to
     *
     * @param lineElements
     * @return RestartFileSection*
     */
    RestartFileSection *RestartFileReader::determineSection(
        std::vector<std::string> &lineElements
    )
    {
        for (const auto &section : _sections)
            if (section->keyword() ==
                utilities::toLowerAndReplaceDashesCopy(lineElements[0]))
                return section.get();

        return _atomSection.get();
    }

    /**
     * @brief Reads a restart file and calls the process function of the
     * corresponding section
     *
     * @throw exc::InputFileException if file not found
     */
    void RestartFileReader::read()
    {
        std::string line;
        size_t      lineNumber = 1;

        while (getline(_fp, line))
        {
            line              = utilities::removeComments(line, "#");
            auto lineElements = utilities::splitString(line);

            if (lineElements.empty())
            {
                ++lineNumber;
                continue;
            }

            auto *section = determineSection(lineElements);
            section->setFilePointer(&_fp);
            section->process(lineElements, _engine, lineNumber);
            ++lineNumber;
        }

        if (_engine.get_simulationBox().getNumberOfAtoms() == 0)
        {
            throw exc::RstFileException(
                std::format(
                    "Start file \"{}\" contains no atoms - a simulation needs "
                    "at least one atom",
                    _fileName
                )
            );
        }

        const auto stepCount = _timingsSettings->getStartingStepCount();

        if (isMDJobType(_engine.getSettings().generalSettings.getJobtype()))
        {
            if (stepCount == 0 && !_timingsSettings->isStartingTimeSet())
            {
                _timingsSettings->setStartingTimeByUser(0 * fs);
            }
            else if (!_timingsSettings->isStartingTimeSet())
            {
                // this entire message is only needed due to a bug we had
                // it was not possible to derive the starting and end time of a
                // simulation only from the number of steps alone it is actually
                // very common to start with a time step of 0.5 fs and later
                // applying some bond constraints to perform simulations at
                // larger time steps (see Shake algorithm) But for certain
                // features of PQ it is necessary to have an accurate starting
                // time to correctly compute cumulative times and related
                // quantities For example for the distance constraints there
                // exist a time dependent force constant that relies on the
                // accurate cumulative time to be correctly applied during all
                // parts of the simulation also after restarting
                const auto message = std::format(
                    "Starting time not set by user while step count is "
                    "non-zero -- total cumulative time not possible to "
                    "be determined. Assuming a homogeneous timestep "
                    "over all previous steps of {}. This problem can be "
                    "resolved "
                    "by providing the starting time in ps explicitly in the "
                    "restart file after the step section e.g.:\n"
                    "step 100 simulation_time 2",
                    _timingsSettings->getTimeStep().in(fs)
                );

                _engine.getLogOutput().writeSetupWarning(message);
                out::StdoutOutput::writeSetupWarning(message);
                _timingsSettings->setStartingTimeByUser(
                    _timingsSettings->getTimeStep() * stepCount
                );
            }
        }

        _writeStepInfo();
    }

    /**
     * @brief wrapper function to construct a RestartFileReader object and call
     * the read function
     *
     * @param engine Reference to the Engine object
     * @param noseHooverSettings Reference to the NoseHooverSettings object
     * @param timingsSettings Reference to the TimingsSettings object
     */
    void readRestartFile(
        engine::Engine     &engine,
        NoseHooverSettings &noseHooverSettings,
        TimingsSettings    &timingsSettings
    )
    {
        const auto &fileSettings = engine.getSettings().fileSettings;

        if (!fileSettings.isStartFileNameSetByUser())
            throw exc::InputFileException(
                "\"start_file\" not set in input file"
            );

        const auto filename = fileSettings.get_startFile();

        out::StdoutOutput::writeRead("Start File", filename);
        auto &logOutput = engine.getLogOutput();
        logOutput.writeRead("Start File", filename);

        RestartFileReader rstFileReader(
            filename,
            engine,
            noseHooverSettings,
            timingsSettings
        );
        rstFileReader.read();
    }

    /**
     * @brief writes the step information to the log file
     *
     */
    void RestartFileReader::_writeStepInfo()
    {
        auto       &logFile = _engine.getLogOutput();
        const auto &jobType =
            _engine.getSettings().generalSettings.getJobtype();

        if (isMDJobType(jobType))
        {
            const auto timeStep  = _timingsSettings->getTimeStep().in(fs);
            const auto startStep = _timingsSettings->getStartingStepCount();
            const auto nSteps    = _timingsSettings->getNumberOfSteps();
            const auto startTime = _timingsSettings->getStartingTime().in(ps);
            const auto deltaTime =
                _timingsSettings->calculateSimulationTimeOfRun().in(ps);
            const auto finalTime =
                _timingsSettings->calculateFinalSimulationTime().in(ps);

            std::string msg;
            msg += std::format("Time Step:              {}", timeStep) + "\n";
            msg += std::format("Starting Step Count:    {}", startStep) + "\n";
            msg += std::format("Number of steps:        {}", nSteps) + "\n";
            msg += std::format("Starting Time:          {}", startTime) + "\n";
            msg += std::format("Simulation Time of Run: {}", deltaTime) + "\n";
            msg += std::format("Total Simulation Time:  {}", finalTime) + "\n";
            logFile.writeSetupInfo(msg);
        }
        else if (isOptJobType(jobType))
        {
            const auto  startStep = _timingsSettings->getStartingStepCount();
            const auto  nSteps    = _timingsSettings->getNumberOfSteps();
            std::string msg;
            msg += std::format(
                       "Starting Step Count:                  {}",
                       startStep
                   ) +
                   "\n";
            msg += std::format(
                       "Maximum Number of optimization steps: {}",
                       nSteps
                   ) +
                   "\n";
            logFile.writeSetupInfo(msg);
        }

        logFile.writeEmptyLine();
    }

}   // namespace input::restartFile
