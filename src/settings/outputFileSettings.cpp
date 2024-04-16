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

#include "outputFileSettings.hpp"

#include <algorithm>   // for for_each
#include <cstdint>     // for UINT64_MAX
#include <string>      // for string, allocator
#include <vector>      // for vector

using settings::OutputFileSettings;

/**
 * @brief Sets the output frequency of the simulation
 *
 * @param outputFreq
 *
 * @throw InputFileException if output frequency is negative
 *
 * @note
 *  if output frequency is 0, it is set to UINT64_MAX
 *  in order to avoid division by 0 in the output
 *
 */
void OutputFileSettings::setOutputFrequency(const size_t outputFreq)
{
    if (0 == outputFreq)
        _outputFrequency = UINT64_MAX;
    else
        _outputFrequency = outputFreq;
}

/**
 * @brief returns the reference file name
 *
 * @details in order to avoid overwriting the reference file and not to set it in the input file - the reference file name is set
 * to the log file name + ".ref"
 *
 * @param restartFileName
 */
std::string OutputFileSettings::getReferenceFileName() { return _logFileName + ".ref"; }

/**
 * @brief sets the file prefix for all output files
 *
 * @param restartFileName
 */
void OutputFileSettings::setFilePrefix(const std::string_view prefix)
{
    _filePrefixSet = true;
    _filePrefix    = prefix;
}

/**
 * @brief replaces the default restart file name
 *
 * @param fileName
 */
void OutputFileSettings::replaceDefaultValues(const std::string &prefix)
{
    if (defaults::_RESTART_FILENAME_DEFAULT_ == _restartFileName)
        _restartFileName = prefix + ".rst";

    if (defaults::_LOG_FILENAME_DEFAULT_ == _logFileName)
        _logFileName = prefix + ".log";

    if (defaults::_TRAJECTORY_FILENAME_DEFAULT_ == _trajectoryFileName)
        _trajectoryFileName = prefix + ".xyz";

    if (defaults::_ENERGY_FILENAME_DEFAULT_ == _energyFileName)
        _energyFileName = prefix + ".en";

    if (defaults::_FORCE_FILENAME_DEFAULT_ == _forceFileName)
        _forceFileName = prefix + ".force";

    if (defaults::_VELOCITY_FILENAME_DEFAULT_ == _velocityFileName)
        _velocityFileName = prefix + ".vel";

    if (defaults::_CHARGE_FILENAME_DEFAULT_ == _chargeFileName)
        _chargeFileName = prefix + ".chrg";

    if (defaults::_INFO_FILENAME_DEFAULT_ == _infoFileName)
        _infoFileName = prefix + ".info";

    if (defaults::_MOMENTUM_FILENAME_DEFAULT_ == _momentumFileName)
        _momentumFileName = prefix + ".mom";

    if (defaults::_VIRIAL_FILENAME_DEFAULT_ == _virialFileName)
        _virialFileName = prefix + ".vir";

    if (defaults::_STRESS_FILENAME_DEFAULT_ == _stressFileName)
        _stressFileName = prefix + ".stress";

    /*****************************
     * ring polymer output files *
     *****************************/

    if (defaults::_RING_POLYMER_RESTART_FILENAME_DEFAULT_ == _ringPolymerRestartFileName)
        _ringPolymerRestartFileName = prefix + ".rpmd.rst";

    if (defaults::_RING_POLYMER_TRAJECTORY_FILENAME_DEFAULT_ == _ringPolymerTrajectoryFileName)
        _ringPolymerTrajectoryFileName = prefix + ".rpmd.xyz";

    if (defaults::_RING_POLYMER_VELOCITY_FILENAME_DEFAULT_ == _ringPolymerVelocityFileName)
        _ringPolymerVelocityFileName = prefix + ".rpmd.vel";

    if (defaults::_RING_POLYMER_FORCE_FILENAME_DEFAULT_ == _ringPolymerForceFileName)
        _ringPolymerForceFileName = prefix + ".rpmd.force";

    if (defaults::_RING_POLYMER_CHARGE_FILENAME_DEFAULT_ == _ringPolymerChargeFileName)
        _ringPolymerChargeFileName = prefix + ".rpmd.chrg";

    if (defaults::_RING_POLYMER_ENERGY_FILENAME_DEFAULT_ == _ringPolymerEnergyFileName)
        _ringPolymerEnergyFileName = prefix + ".rpmd.en";
}

/**
 * @brief determines the most common prefix of all output files
 *
 * @return most common prefix
 */
std::string OutputFileSettings::determineMostCommonPrefix()
{

    std::vector<std::string> fileNames = {_restartFileName,
                                          _logFileName,
                                          _trajectoryFileName,
                                          _energyFileName,
                                          _forceFileName,
                                          _velocityFileName,
                                          _chargeFileName,
                                          _infoFileName,
                                          _momentumFileName,
                                          _virialFileName,
                                          _stressFileName,
                                          _ringPolymerRestartFileName,
                                          _ringPolymerTrajectoryFileName,
                                          _ringPolymerVelocityFileName,
                                          _ringPolymerForceFileName,
                                          _ringPolymerChargeFileName};

    auto removeEnding = [](std::string &fileName)
    {
        const auto pos = fileName.find_first_of('.');
        if (pos != std::string::npos)
            fileName.erase(pos);
    };

    std::ranges::for_each(fileNames, removeEnding);

    auto uniqueFileNames = fileNames;

    std::ranges::sort(uniqueFileNames);
    const auto [first, last] = std::ranges::unique(uniqueFileNames);
    uniqueFileNames.erase(first, last);

    std::string mostCommonPrefix = "default";
    int         count            = 0;

    auto getHighestOccurrence = [&fileNames, &mostCommonPrefix, &count](const std::string &fileName)
    {
        if (fileName == "default")
            return;

        const int occurrence = std::ranges::count(fileNames, fileName);
        if (occurrence > count)
        {
            mostCommonPrefix = fileName;
            count            = occurrence;
        }
    };

    std::ranges::for_each(uniqueFileNames, getHighestOccurrence);

    return mostCommonPrefix;
}
