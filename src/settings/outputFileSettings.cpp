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

    if (!isFilePrefixSet())
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
void OutputFileSettings::setOutputFrequency(size_t outputFreq)
{
    if (0 == outputFreq)
        _outputFrequency.set(UINT64_MAX);
    else
        _outputFrequency.set(outputFreq);
}

/**
 * @brief sets the file prefix for all output files
 *
 * @param prefix
 */
void OutputFileSettings::setFilePrefix(const std::string& prefix)
{
    _filePrefix.set(prefix);
}

/**
 * @brief replaces the default restart file name
 *
 * @param prefix
 */
void OutputFileSettings::_replaceDefaultValues(const std::string& prefix)
{
    if (DefaultFiles::restartFile == _rstFile.get())
        _rstFile.set(prefix + ".rst");

    if (DefaultFiles::logFile == _logFile.get())
        _logFile.set(prefix + ".log");

    if (DefaultFiles::refFile == _refFile.get())
        _refFile.set(prefix + ".ref");

    if (DefaultFiles::trajFile == _trajFile.get())
        _trajFile.set(prefix + ".xyz");

    if (DefaultFiles::hybridCenterFile == _hybridCenterFile.get())
        _hybridCenterFile.set(prefix + ".center.xyz");

    if (DefaultFiles::energyFile == _energyFile.get())
        _energyFile.set(prefix + ".en");

    if (DefaultFiles::instEnFile == _instEnFile.get())
        _instEnFile.set(prefix + ".instant_en");

    if (DefaultFiles::forceFile == _forceFile.get())
        _forceFile.set(prefix + ".force");

    if (DefaultFiles::velFile == _velFile.get())
        _velFile.set(prefix + ".vel");

    if (DefaultFiles::chargeFile == _chargeFile.get())
        _chargeFile.set(prefix + ".chrg");

    if (DefaultFiles::infoFile == _infoFile.get())
        _infoFile.set(prefix + ".info");

    if (DefaultFiles::momentumFile == _momFile.get())
        _momFile.set(prefix + ".mom");

    if (DefaultFiles::virialFile == _virialFile.get())
        _virialFile.set(prefix + ".vir");

    if (DefaultFiles::stressFile == _stressFile.get())
        _stressFile.set(prefix + ".stress");

    if (DefaultFiles::boxFile == _boxFile.get())
        _boxFile.set(prefix + ".box");

    if (DefaultFiles::optFile == _optFile.get())
        _optFile.set(prefix + ".opt");

    /*****************************
     * ring polymer output files *
     *****************************/

    if (DefaultFiles::rpmdRstFile == _rpmdRstFile.get())
        _rpmdRstFile.set(prefix + ".rpmd.rst");

    if (DefaultFiles::rpmdTrajFile == _rpmdTrajFile.get())
        _rpmdTrajFile.set(prefix + ".rpmd.xyz");

    if (DefaultFiles::rpmdVelFile == _rpmdVelFile.get())
        _rpmdVelFile.set(prefix + ".rpmd.vel");

    if (DefaultFiles::rpmdForceFile == _rpmdForceFile.get())
        _rpmdForceFile.set(prefix + ".rpmd.force");

    if (DefaultFiles::rpmdChargeFile == _rpmdChargeFile.get())
        _rpmdChargeFile.set(prefix + ".rpmd.chrg");

    if (DefaultFiles::rpmdEnergyFile == _rpmdEnergyFile.get())
        _rpmdEnergyFile.set(prefix + ".rpmd.en");

    /********************
     * the timings file *
     ********************/

    if (DefaultFiles::timingsFile == _timeFile.get())
        _timeFile.set(prefix + ".timings");
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

/***************************
 *                         *
 * standard setter methods *
 *                         *
 ***************************/

/**
 * @brief sets the restart file name
 *
 * @param name
 */
void OutputFileSettings::setRestartFileName(const std::string& name)
{
    _rstFile.set(name);
}

/**
 * @brief sets the energy file name
 *
 * @param name
 */
void OutputFileSettings::setEnergyFileName(const std::string& name)
{
    _energyFile.set(name);
}

/**
 * @brief sets the instant energy file name
 *
 * @param name
 */
void OutputFileSettings::setInstantEnergyFileName(const std::string& name)
{
    _instEnFile.set(name);
}
/**
 * @brief sets the momentum file name
 *
 * @param name
 */
void OutputFileSettings::setMomentumFileName(const std::string& name)
{
    _momFile.set(name);
}

/**
 * @brief sets the trajectory file name
 *
 * @param name
 */
void OutputFileSettings::setTrajectoryFileName(const std::string& name)
{
    _trajFile.set(name);
}

/**
 * @brief sets the hybrid center file name
 *
 * @param name
 */
void OutputFileSettings::setHybridCenterFileName(const std::string& name)
{
    _hybridCenterFile.set(name);
}

/**
 * @brief sets the velocity file name
 *
 * @param name
 */
void OutputFileSettings::setVelocityFileName(const std::string& name)
{
    _velFile.set(name);
}

/**
 * @brief sets the force file name
 *
 * @param name
 */
void OutputFileSettings::setForceFileName(const std::string& name)
{
    _forceFile.set(name);
}

/**
 * @brief sets the charge file name
 *
 * @param name
 */
void OutputFileSettings::setChargeFileName(const std::string& name)
{
    _chargeFile.set(name);
}

/**
 * @brief sets the log file name
 *
 * @param name
 */
void OutputFileSettings::setLogFileName(const std::string& name)
{
    _logFile.set(name);
}

/**
 * @brief sets the ref file name
 *
 * @param name
 */
void OutputFileSettings::setRefFileName(const std::string& name)
{
    _refFile.set(name);
}

/**
 * @brief sets the info file name
 *
 * @param name
 */
void OutputFileSettings::setInfoFileName(const std::string& name)
{
    _infoFile.set(name);
}

/**
 * @brief sets the virial file name
 *
 * @param name
 */
void OutputFileSettings::setVirialFileName(const std::string& name)
{
    _virialFile.set(name);
}

/**
 * @brief sets the stress file name
 *
 * @param name
 */
void OutputFileSettings::setStressFileName(const std::string& name)
{
    _stressFile.set(name);
}

/**
 * @brief sets the box file name
 *
 * @param name
 */
void OutputFileSettings::setBoxFileName(const std::string& name)
{
    _boxFile.set(name);
}

/**
 * @brief sets the optimization file name
 *
 * @param name
 */
void OutputFileSettings::setOptFileName(const std::string& name)
{
    _optFile.set(name);
}

/**
 * @brief sets the ring polymer restart file name
 *
 * @param name
 */
void OutputFileSettings::setRingPolymerRestartFileName(const std::string& name)
{
    _rpmdRstFile.set(name);
}

/**
 * @brief sets the ring polymer trajectory file name
 *
 * @param name
 */
void OutputFileSettings::setRingPolymerTrajectoryFileName(
    const std::string& name
)
{
    _rpmdTrajFile.set(name);
}

/**
 * @brief sets the ring polymer velocity file name
 *
 * @param name
 */
void OutputFileSettings::setRingPolymerVelocityFileName(const std::string& name)
{
    _rpmdVelFile.set(name);
}

/**
 * @brief sets the ring polymer force file name
 *
 * @param name
 */
void OutputFileSettings::setRingPolymerForceFileName(const std::string& name)
{
    _rpmdForceFile.set(name);
}

/**
 * @brief sets the ring polymer charge file name
 *
 * @param name
 */
void OutputFileSettings::setRingPolymerChargeFileName(const std::string& name)
{
    _rpmdChargeFile.set(name);
}

/**
 * @brief sets the ring polymer energy file name
 *
 * @param name
 */
void OutputFileSettings::setRingPolymerEnergyFileName(const std::string& name)
{
    _rpmdEnergyFile.set(name);
}

/**
 * @brief sets the timings file name
 *
 * @param name
 */
void OutputFileSettings::setTimingsFileName(const std::string& name)
{
    _timeFile.set(name);
}

/**
 * @brief sets if existing output files will be overwritten
 *
 * @param overwrite
 */
void OutputFileSettings::setOverwriteOutputFiles(bool overwrite)
{
    _overwriteOutputFiles.set(overwrite);
}

/**
 * @brief sets if output files should include metadata
 *
 * @param includeMetadata
 */
void OutputFileSettings::setIncludeOutputMetadata(bool includeMetadata)
{
    _includeOutputMetadata.set(includeMetadata);
}

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

/**
 * @brief get the output frequency
 *
 * @return size_t
 */
size_t OutputFileSettings::getOutputFrequency() const
{
    return _outputFrequency.get();
}

/**
 * @brief determine if the file prefix is set
 *
 * @return std::string
 */
bool OutputFileSettings::isFilePrefixSet() const { return _filePrefix.isSet(); }

/**
 * @brief get the file prefix
 *
 * @return std::string
 */
const std::string& OutputFileSettings::getFilePrefix() const
{
    return _filePrefix.get();
}

/**
 * @brief get the restart file name
 *
 * @return std::string
 */
const std::string& OutputFileSettings::getRestartFileName() const
{
    return _rstFile.get();
}

/**
 * @brief get the energy file name
 *
 * @return std::string
 */
const std::string& OutputFileSettings::getEnergyFileName() const
{
    return _energyFile.get();
}

/**
 * @brief get the instant energy file name
 *
 * @return std::string
 */
const std::string& OutputFileSettings::getInstantEnergyFileName() const
{
    return _instEnFile.get();
}

/**
 * @brief get the momentum file name
 *
 * @return std::string
 */
const std::string& OutputFileSettings::getMomentumFileName() const
{
    return _momFile.get();
}

/**
 * @brief get the trajectory file name
 *
 * @return std::string
 */
const std::string& OutputFileSettings::getTrajectoryFileName() const
{
    return _trajFile.get();
}

/**
 * @brief get the hybrid center file name
 *
 * @return std::string
 */
const std::string& OutputFileSettings::getHybridCenterFileName() const
{
    return _hybridCenterFile.get();
}

/**
 * @brief get the velocity file name
 *
 * @return std::string
 */
const std::string& OutputFileSettings::getVelocityFileName() const
{
    return _velFile.get();
}

/**
 * @brief get the force file name
 *
 * @return std::string
 */
const std::string& OutputFileSettings::getForceFileName() const
{
    return _forceFile.get();
}

/**
 * @brief get the charge file name
 *
 * @return std::string
 */
const std::string& OutputFileSettings::getChargeFileName() const
{
    return _chargeFile.get();
}

/**
 * @brief get the log file name
 *
 * @return std::string
 */
const std::string& OutputFileSettings::getLogFileName() const
{
    return _logFile.get();
}

/**
 * @brief get the ref file name
 *
 * @return std::string
 */
const std::string& OutputFileSettings::getRefFileName() const
{
    return _refFile.get();
}

/**
 * @brief get the info file name
 *
 * @return std::string
 */
const std::string& OutputFileSettings::getInfoFileName() const
{
    return _infoFile.get();
}

/**
 * @brief get the virial file name
 *
 * @return std::string
 */
const std::string& OutputFileSettings::getVirialFileName() const
{
    return _virialFile.get();
}

/**
 * @brief get the stress file name
 *
 * @return std::string
 */
const std::string& OutputFileSettings::getStressFileName() const
{
    return _stressFile.get();
}

/**
 * @brief get the box file name
 *
 * @return std::string
 */
const std::string& OutputFileSettings::getBoxFileName() const
{
    return _boxFile.get();
}

/**
 * @brief get the optimization file name
 *
 * @return std::string
 */
const std::string& OutputFileSettings::getOptFileName() const
{
    return _optFile.get();
}

/**
 * @brief get the ring polymer restart file name
 *
 * @return std::string
 */
const std::string& OutputFileSettings::getRPMDRestartFileName() const
{
    return _rpmdRstFile.get();
}

/**
 * @brief get the ring polymer trajectory file name
 *
 * @return std::string
 */
const std::string& OutputFileSettings::getRPMDTrajFileName() const
{
    return _rpmdTrajFile.get();
}

/**
 * @brief get the ring polymer velocity file name
 *
 * @return std::string
 */
const std::string& OutputFileSettings::getRPMDVelocityFileName() const
{
    return _rpmdVelFile.get();
}

/**
 * @brief get the ring polymer force file name
 *
 * @return std::string
 */
const std::string& OutputFileSettings::getRPMDForceFileName() const
{
    return _rpmdForceFile.get();
}

/**
 * @brief get the ring polymer charge file name
 *
 * @return std::string
 */
const std::string& OutputFileSettings::getRPMDChargeFileName() const
{
    return _rpmdChargeFile.get();
}

/**
 * @brief get the ring polymer energy file name
 *
 * @return std::string
 */
const std::string& OutputFileSettings::getRPMDEnergyFileName() const
{
    return _rpmdEnergyFile.get();
}

/**
 * @brief get the timings file name
 *
 * @return std::string
 */
const std::string& OutputFileSettings::getTimingsFileName() const
{
    return _timeFile.get();
}

/**
 * @brief returns if existing output files will be overwritten
 *
 * @return bool
 */
bool OutputFileSettings::getOverwriteOutputFiles() const
{
    return _overwriteOutputFiles.get();
}

/**
 * @brief returns if output files should include metadata
 *
 * @return bool
 */
bool OutputFileSettings::getIncludeOutputMetadata() const
{
    return _includeOutputMetadata.get();
}
