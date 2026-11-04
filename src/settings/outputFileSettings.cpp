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
 * @brief sets the file prefix for all output files
 *
 * @param prefix
 */
void OutputFileSettings::setFilePrefixByUser(const std::string& prefix)
{
    _filePrefix.setByUser(prefix);
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
void OutputFileSettings::setRestartFileNameByUser(const std::string& name)
{
    _rstFile.setByUser(name);
}

/**
 * @brief sets the energy file name
 *
 * @param name
 */
void OutputFileSettings::setEnergyFileNameByUser(const std::string& name)
{
    _energyFile.setByUser(name);
}

/**
 * @brief sets the instant energy file name
 *
 * @param name
 */
void OutputFileSettings::setInstantEnergyFileNameByUser(const std::string& name)
{
    _instEnFile.setByUser(name);
}
/**
 * @brief sets the momentum file name
 *
 * @param name
 */
void OutputFileSettings::setMomentumFileNameByUser(const std::string& name)
{
    _momFile.setByUser(name);
}

/**
 * @brief sets the trajectory file name
 *
 * @param name
 */
void OutputFileSettings::setTrajectoryFileNameByUser(const std::string& name)
{
    _trajFile.setByUser(name);
}

/**
 * @brief sets the hybrid center file name
 *
 * @param name
 */
void OutputFileSettings::setHybridCenterFileNameByUser(const std::string& name)
{
    _hybridCenterFile.setByUser(name);
}

/**
 * @brief sets the velocity file name
 *
 * @param name
 */
void OutputFileSettings::setVelocityFileNameByUser(const std::string& name)
{
    _velFile.setByUser(name);
}

/**
 * @brief sets the force file name
 *
 * @param name
 */
void OutputFileSettings::setForceFileNameByUser(const std::string& name)
{
    _forceFile.setByUser(name);
}

/**
 * @brief sets the charge file name
 *
 * @param name
 */
void OutputFileSettings::setChargeFileNameByUser(const std::string& name)
{
    _chargeFile.setByUser(name);
}

/**
 * @brief sets the log file name
 *
 * @param name
 */
void OutputFileSettings::setLogFileNameByUser(const std::string& name)
{
    _logFile.setByUser(name);
}

/**
 * @brief sets the ref file name
 *
 * @param name
 */
void OutputFileSettings::setRefFileNameByUser(const std::string& name)
{
    _refFile.setByUser(name);
}

/**
 * @brief sets the info file name
 *
 * @param name
 */
void OutputFileSettings::setInfoFileNameByUser(const std::string& name)
{
    _infoFile.setByUser(name);
}

/**
 * @brief sets the virial file name
 *
 * @param name
 */
void OutputFileSettings::setVirialFileNameByUser(const std::string& name)
{
    _virialFile.setByUser(name);
}

/**
 * @brief sets the stress file name
 *
 * @param name
 */
void OutputFileSettings::setStressFileNameByUser(const std::string& name)
{
    _stressFile.setByUser(name);
}

/**
 * @brief sets the box file name
 *
 * @param name
 */
void OutputFileSettings::setBoxFileNameByUser(const std::string& name)
{
    _boxFile.setByUser(name);
}

/**
 * @brief sets the optimization file name
 *
 * @param name
 */
void OutputFileSettings::setOptFileNameByUser(const std::string& name)
{
    _optFile.setByUser(name);
}

/**
 * @brief sets the ring polymer restart file name
 *
 * @param name
 */
void OutputFileSettings::setRingPolymerRestartFileNameByUser(
    const std::string& name
)
{
    _rpmdRstFile.setByUser(name);
}

/**
 * @brief sets the ring polymer trajectory file name
 *
 * @param name
 */
void OutputFileSettings::setRingPolymerTrajectoryFileNameByUser(
    const std::string& name
)
{
    _rpmdTrajFile.setByUser(name);
}

/**
 * @brief sets the ring polymer velocity file name
 *
 * @param name
 */
void OutputFileSettings::setRingPolymerVelocityFileNameByUser(
    const std::string& name
)
{
    _rpmdVelFile.setByUser(name);
}

/**
 * @brief sets the ring polymer force file name
 *
 * @param name
 */
void OutputFileSettings::setRingPolymerForceFileNameByUser(
    const std::string& name
)
{
    _rpmdForceFile.setByUser(name);
}

/**
 * @brief sets the ring polymer charge file name
 *
 * @param name
 */
void OutputFileSettings::setRingPolymerChargeFileNameByUser(
    const std::string& name
)
{
    _rpmdChargeFile.setByUser(name);
}

/**
 * @brief sets the ring polymer energy file name
 *
 * @param name
 */
void OutputFileSettings::setRingPolymerEnergyFileNameByUser(
    const std::string& name
)
{
    _rpmdEnergyFile.setByUser(name);
}

/**
 * @brief sets the timings file name
 *
 * @param name
 */
void OutputFileSettings::setTimingsFileNameByUser(const std::string& name)
{
    _timeFile.setByUser(name);
}

/**
 * @brief sets if existing output files will be overwritten
 *
 * @param overwrite
 */
void OutputFileSettings::setOverwriteOutputFilesByUser(bool overwrite)
{
    _overwriteOutputFiles.setByUser(overwrite);
}

/**
 * @brief sets if output files should include metadata
 *
 * @param includeMetadata
 */
void OutputFileSettings::setIncludeOutputMetadataByUser(bool includeMetadata)
{
    _includeOutputMetadata.setByUser(includeMetadata);
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
