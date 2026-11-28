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

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "defaults.hpp"

/**
 * @brief finalizes the output file settings by determining the file prefix
 *        and replacing default file names with the appropriate prefixed names
 */
void OutputFileSettings::finalize()
{
    std::string prefix;

    if (!_filePrefix.isSetByUser())
        prefix = _determineMostCommonPrefix();
    else
        prefix = _filePrefix.get();

    _replaceDefaultValues(prefix);
}

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
void OutputFileSettings::setOutputFrequencyByUser(size_t outputFreq)
{
    if (0 == outputFreq)
        _outputFrequency.setByUser(UINT64_MAX);
    else
        _outputFrequency.setByUser(outputFreq);
}

/**
 * @brief replaces the default restart file name
 *
 * @param prefix
 */
void OutputFileSettings::_replaceDefaultValues(const std::string& prefix)
{
    if (!_rstFile.isSetByUser())
        _rstFile.setBySystem(prefix + ".rst");

    if (!_logFile.isSetByUser())
        _logFile.setBySystem(prefix + ".log");

    if (!_refFile.isSetByUser())
        _refFile.setBySystem(prefix + ".ref");

    if (!_trajFile.isSetByUser())
        _trajFile.setBySystem(prefix + ".xyz");

    if (!_extxyzFile.isSetByUser())
        _extxyzFile.setBySystem(prefix + ".extxyz");

    if (!_h5mdFile.isSetByUser())
        _h5mdFile.setBySystem(prefix + ".h5md");

    if (!_aseTrajFile.isSetByUser())
    {
        _aseTrajFile.setBySystem(prefix + ".ase.traj");
    }

    if (!_hybridCenterFile.isSetByUser())
        _hybridCenterFile.setBySystem(prefix + ".center.xyz");

    if (!_energyFile.isSetByUser())
        _energyFile.setBySystem(prefix + ".en");

    if (!_instEnFile.isSetByUser())
        _instEnFile.setBySystem(prefix + ".instant_en");

    if (!_forceFile.isSetByUser())
        _forceFile.setBySystem(prefix + ".force");

    if (!_velFile.isSetByUser())
        _velFile.setBySystem(prefix + ".vel");

    if (!_chargeFile.isSetByUser())
        _chargeFile.setBySystem(prefix + ".chrg");

    if (!_infoFile.isSetByUser())
        _infoFile.setBySystem(prefix + ".info");

    if (!_momFile.isSetByUser())
        _momFile.setBySystem(prefix + ".mom");

    if (!_virialFile.isSetByUser())
        _virialFile.setBySystem(prefix + ".vir");

    if (!_stressFile.isSetByUser())
        _stressFile.setBySystem(prefix + ".stress");

    if (!_boxFile.isSetByUser())
        _boxFile.setBySystem(prefix + ".box");

    if (!_optFile.isSetByUser())
        _optFile.setBySystem(prefix + ".opt");

    /*****************************
     * ring polymer output files *
     *****************************/

    if (!_rpmdRstFile.isSetByUser())
        _rpmdRstFile.setBySystem(prefix + ".rpmd.rst");

    if (!_rpmdTrajFile.isSetByUser())
        _rpmdTrajFile.setBySystem(prefix + ".rpmd.xyz");

    if (!_rpmdVelFile.isSetByUser())
        _rpmdVelFile.setBySystem(prefix + ".rpmd.vel");

    if (!_rpmdForceFile.isSetByUser())
        _rpmdForceFile.setBySystem(prefix + ".rpmd.force");

    if (!_rpmdChargeFile.isSetByUser())
        _rpmdChargeFile.setBySystem(prefix + ".rpmd.chrg");

    if (!_rpmdEnergyFile.isSetByUser())
        _rpmdEnergyFile.setBySystem(prefix + ".rpmd.en");

    /********************
     * the timings file *
     ********************/

    if (!_timeFile.isSetByUser())
        _timeFile.setBySystem(prefix + ".timings");
}

/**
 * @brief determines the most common prefix of all output files
 *
 * @return most common prefix
 */
std::string OutputFileSettings::_determineMostCommonPrefix() const
{
    std::vector<std::string> fileNames = {
        _rstFile.get(),       _logFile.get(),        _trajFile.get(),
        _energyFile.get(),    _instEnFile.get(),     _forceFile.get(),
        _velFile.get(),       _chargeFile.get(),     _infoFile.get(),
        _momFile.get(),

        _virialFile.get(),    _stressFile.get(),     _boxFile.get(),
        _optFile.get(),

        _rpmdRstFile.get(),   _rpmdTrajFile.get(),   _rpmdVelFile.get(),
        _rpmdForceFile.get(), _rpmdChargeFile.get(),

        _timeFile.get()
    };

    if (_extxyzFile.isSetByUser())
        fileNames.push_back(_extxyzFile.get());

    if (_h5mdFile.isSetByUser())
        fileNames.push_back(_h5mdFile.get());

    if (_aseTrajFile.isSetByUser())
    {
        fileNames.push_back(_aseTrajFile.get());
    }

    auto removeEnding = [](std::string& fileName)
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

    std::string mostCommonPrefix = DefaultFiles::prefix;
    auto        count            = 0;

    auto getHighestOccurrence =
        [&fileNames, &mostCommonPrefix, &count](const std::string& fileName)
    {
        if (fileName == DefaultFiles::prefix)
            return;

        const auto occurrence = std::ranges::count(fileNames, fileName);

        if (occurrence > count)
        {
            mostCommonPrefix = fileName;
            count            = static_cast<int>(occurrence);
        }
    };

    std::ranges::for_each(uniqueFileNames, getHighestOccurrence);

    return mostCommonPrefix;
}
