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
#include "exceptions.hpp"
#include "logOutput.hpp"
#include "metaDataSection.hpp"
#include "noseHooverSection.hpp"
#include "settings.hpp"
#include "simulationBox.hpp"
#include "stdoutOutput.hpp"
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
     */
    RestartFileReader::RestartFileReader(const std::string &filename)
        : _fileName(filename),
          _fp(filename),
          _atomSection(std::make_unique<AtomSection>())
    {
        _sections.push_back(std::make_unique<BoxSection>());
        _sections.push_back(std::make_unique<NoseHooverSection>());
        _sections.push_back(std::make_unique<StepCountSection>());
        _sections.push_back(std::make_unique<MetaDataSection>());
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
    void RestartFileReader::read(
        Settings              &settings,
        molsys::SimulationBox &simulationBox,
        out::LogOutput        &logOutput
    )
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
            section->process(lineElements, lineNumber, settings, simulationBox);
            ++lineNumber;
        }

        if (simulationBox.getNumberOfAtoms() == 0)
        {
            throw exc::RstFileException(
                std::format(
                    "Start file \"{}\" contains no atoms - a simulation needs "
                    "at least one atom",
                    _fileName
                )
            );
        }

        const auto stepCount = settings.timings.getStartingStepCount();

        if (isMDJobType(settings.generalSettings.getJobtype()))
        {
            if (stepCount == 0 && !settings.timings.isStartingTimeSet())
            {
                settings.timings.setStartingTimeByUser(0 * fs);
            }
            else if (!settings.timings.isStartingTimeSet())
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
                    settings.timings.getTimeStep().in(fs)
                );

                logOutput.writeSetupWarning(message);
                out::StdoutOutput::writeSetupWarning(message);
                settings.timings.setStartingTimeByUser(
                    settings.timings.getTimeStep() * stepCount
                );
            }
        }

        _writeStepInfo(settings, logOutput);
    }

    /**
     * @brief wrapper function to construct a RestartFileReader object and call
     * the read function
     *
     */
    void readRestartFile(
        Settings              &settings,
        molsys::SimulationBox &simulationBox,
        out::LogOutput        &logOutput
    )
    {
        if (!settings.fileSettings.isStartFileNameSetByUser())
            throw exc::InputFileException(
                "\"start_file\" not set in input file"
            );

        const auto filename = settings.fileSettings.get_startFile();

        out::StdoutOutput::writeRead("Start File", filename);
        logOutput.writeRead("Start File", filename);

        RestartFileReader rstFileReader(filename);
        rstFileReader.read(settings, simulationBox, logOutput);
    }

    /**
     * @brief writes the step information to the log file
     *
     */
    void RestartFileReader::_writeStepInfo(
        const Settings &settings,
        out::LogOutput &logOutput
    )
    {
        const auto &jobType = settings.generalSettings.getJobtype();

        if (isMDJobType(jobType))
        {
            const auto timeStep  = settings.timings.getTimeStep().in(fs);
            const auto startStep = settings.timings.getStartingStepCount();
            const auto nSteps    = settings.timings.getNumberOfSteps();
            const auto startTime = settings.timings.getStartingTime().in(ps);
            const auto deltaTime =
                settings.timings.calculateSimulationTimeOfRun().in(ps);
            const auto finalTime =
                settings.timings.calculateFinalSimulationTime().in(ps);

            std::string msg;
            msg += std::format("Time Step:              {}", timeStep) + "\n";
            msg += std::format("Starting Step Count:    {}", startStep) + "\n";
            msg += std::format("Number of steps:        {}", nSteps) + "\n";
            msg += std::format("Starting Time:          {}", startTime) + "\n";
            msg += std::format("Simulation Time of Run: {}", deltaTime) + "\n";
            msg += std::format("Total Simulation Time:  {}", finalTime) + "\n";
            logOutput.writeSetupInfo(msg);
        }
        else if (isOptJobType(jobType))
        {
            const auto  startStep = settings.timings.getStartingStepCount();
            const auto  nSteps    = settings.timings.getNumberOfSteps();
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
            logOutput.writeSetupInfo(msg);
        }

        logOutput.writeEmptyLine();
    }

}   // namespace input::restartFile
