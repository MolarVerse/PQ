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
 * @details in order to avoid overwriting the reference file and not to set it
 * in the input file - the reference file name is set to the log file name +
 * ".ref"
 *
 * @param restartFileName
 */
std::string OutputFileSettings::getReferenceFileName()
{
    return _logFile + ".ref";
}

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
    if (defaults::_RESTART_FILE_DEFAULT_ == _rstFile)
        _rstFile = prefix + ".rst";

    if (defaults::_LOG_FILE_DEFAULT_ == _logFile)
        _logFile = prefix + ".log";

    if (defaults::_TRAJ_FILE_DEFAULT_ == _trajFile)
        _trajFile = prefix + ".xyz";

    if (defaults::_ENERGY_FILE_DEFAULT_ == _energyFile)
        _energyFile = prefix + ".en";

    if (defaults::_INSTEN_FILE_DEFAULT_ == _instEnFile)
        _instEnFile = prefix + ".instant_en";

    if (defaults::_FORCE_FILE_DEFAULT_ == _forceFile)
        _forceFile = prefix + ".force";

    if (defaults::_VEL_FILE_DEFAULT_ == _velFile)
        _velFile = prefix + ".vel";

    if (defaults::_CHARGE_FILE_DEFAULT_ == _chargeFile)
        _chargeFile = prefix + ".chrg";

    if (defaults::_INFO_FILE_DEFAULT_ == _infoFile)
        _infoFile = prefix + ".info";

    if (defaults::_MOMENTUM_FILE_DEFAULT_ == _momFile)
        _momFile = prefix + ".mom";

    if (defaults::_VIRIAL_FILE_DEFAULT_ == _virialFile)
        _virialFile = prefix + ".vir";

    if (defaults::_STRESS_FILE_DEFAULT_ == _stressFile)
        _stressFile = prefix + ".stress";

    if (defaults::_BOX_FILE_DEFAULT_ == _boxFile)
        _boxFile = prefix + ".box";

    if (defaults::_OPT_FILE_DEFAULT_ == _optFile)
        _optFile = prefix + ".opt";

    /*****************************
     * ring polymer output files *
     *****************************/

    if (defaults::_RPMD_RST_FILE_DEFAULT_ == _rpmdRstFile)
        _rpmdRstFile = prefix + ".rpmd.rst";

    if (defaults::_RPMD_TRAJ_FILE_DEFAULT_ == _rpmdTrajFile)
        _rpmdTrajFile = prefix + ".rpmd.xyz";

    if (defaults::_RPMD_VEL_FILE_DEFAULT_ == _rpmdVelFile)
        _rpmdVelFile = prefix + ".rpmd.vel";

    if (defaults::_RPMD_FORCE_FILE_DEFAULT_ == _rpmdForceFile)
        _rpmdForceFile = prefix + ".rpmd.force";

    if (defaults::_RPMD_CHARGE_FILE_DEFAULT_ == _rpmdChargeFile)
        _rpmdChargeFile = prefix + ".rpmd.chrg";

    if (defaults::_RPMD_ENERGY_FILE_DEFAULT_ == _rpmdEnergyFile)
        _rpmdEnergyFile = prefix + ".rpmd.en";

    /********************
     * the timings file *
     ********************/

    if (defaults::_TIMINGS_FILE_DEFAULT_ == _timeFile)
        _timeFile = prefix + ".timings";
}

/**
 * @brief determines the most common prefix of all output files
 *
 * @return most common prefix
 */
std::string OutputFileSettings::determineMostCommonPrefix()
{
    std::vector<std::string> fileNames = {
        _rstFile,        _logFile,      _trajFile,    _energyFile,
        _instEnFile,     _forceFile,    _velFile,     _chargeFile,
        _infoFile,       _momFile,

        _virialFile,     _stressFile,   _boxFile,     _optFile,

        _rpmdRstFile,    _rpmdTrajFile, _rpmdVelFile, _rpmdForceFile,
        _rpmdChargeFile,

        _timeFile
    };

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

    auto getHighestOccurrence =
        [&fileNames, &mostCommonPrefix, &count](const std::string &fileName)
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
void OutputFileSettings::setRestartFileName(const std::string_view name)
{
    _rstFile = name;
}

/**
 * @brief sets the energy file name
 *
 * @param name
 */
void OutputFileSettings::setEnergyFileName(const std::string_view name)
{
    _energyFile = name;
}

/**
 * @brief sets the instant energy file name
 *
 * @param name
 */
void OutputFileSettings::setInstantEnergyFileName(const std::string_view name)
{
    _instEnFile = name;
}
/**
 * @brief sets the momentum file name
 *
 * @param name
 */
void OutputFileSettings::setMomentumFileName(const std::string_view name)
{
    _momFile = name;
}

/**
 * @brief sets the trajectory file name
 *
 * @param name
 */
void OutputFileSettings::setTrajectoryFileName(const std::string_view name)
{
    _trajFile = name;
}

/**
 * @brief sets the velocity file name
 *
 * @param name
 */
void OutputFileSettings::setVelocityFileName(const std::string_view name)
{
    _velFile = name;
}

/**
 * @brief sets the force file name
 *
 * @param name
 */
void OutputFileSettings::setForceFileName(const std::string_view name)
{
    _forceFile = name;
}

/**
 * @brief sets the charge file name
 *
 * @param name
 */
void OutputFileSettings::setChargeFileName(const std::string_view name)
{
    _chargeFile = name;
}

/**
 * @brief sets the log file name
 *
 * @param name
 */
void OutputFileSettings::setLogFileName(const std::string_view name)
{
    _logFile = name;
}

/**
 * @brief sets the info file name
 *
 * @param name
 */
void OutputFileSettings::setInfoFileName(const std::string_view name)
{
    _infoFile = name;
}

/**
 * @brief sets the virial file name
 *
 * @param name
 */
void OutputFileSettings::setVirialFileName(const std::string_view name)
{
    _virialFile = name;
}

/**
 * @brief sets the stress file name
 *
 * @param name
 */
void OutputFileSettings::setStressFileName(const std::string_view name)
{
    _stressFile = name;
}

/**
 * @brief sets the box file name
 *
 * @param name
 */
void OutputFileSettings::setBoxFileName(const std::string_view name)
{
    _boxFile = name;
}

/**
 * @brief sets the optimization file name
 *
 * @param name
 */
void OutputFileSettings::setOptFileName(const std::string_view name)
{
    _optFile = name;
}

/**
 * @brief sets the ring polymer restart file name
 *
 * @param name
 */
void OutputFileSettings::setRingPolymerRestartFileName(
    const std::string_view name
)
{
    _rpmdRstFile = name;
}

/**
 * @brief sets the ring polymer trajectory file name
 *
 * @param name
 */
void OutputFileSettings::setRingPolymerTrajectoryFileName(
    const std::string_view name
)
{
    _rpmdTrajFile = name;
}

/**
 * @brief sets the ring polymer velocity file name
 *
 * @param name
 */
void OutputFileSettings::setRingPolymerVelocityFileName(
    const std::string_view name
)
{
    _rpmdVelFile = name;
}

/**
 * @brief sets the ring polymer force file name
 *
 * @param name
 */
void OutputFileSettings::setRingPolymerForceFileName(const std::string_view name
)
{
    _rpmdForceFile = name;
}

/**
 * @brief sets the ring polymer charge file name
 *
 * @param name
 */
void OutputFileSettings::setRingPolymerChargeFileName(
    const std::string_view name
)
{
    _rpmdChargeFile = name;
}

/**
 * @brief sets the ring polymer energy file name
 *
 * @param name
 */
void OutputFileSettings::setRingPolymerEnergyFileName(
    const std::string_view name
)
{
    _rpmdEnergyFile = name;
}

/**
 * @brief sets the timings file name
 *
 * @param name
 */
void OutputFileSettings::setTimingsFileName(const std::string_view name)
{
    _timeFile = name;
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
size_t OutputFileSettings::getOutputFrequency() { return _outputFrequency; }

/**
 * @brief determine if the file prefix is set
 *
 * @return std::string
 */
bool OutputFileSettings::isFilePrefixSet() { return _filePrefixSet; }

/**
 * @brief get the file prefix
 *
 * @return std::string
 */
std::string OutputFileSettings::getFilePrefix() { return _filePrefix; }

/**
 * @brief get the restart file name
 *
 * @return std::string
 */
std::string OutputFileSettings::getRestartFileName() { return _rstFile; }

/**
 * @brief get the energy file name
 *
 * @return std::string
 */
std::string OutputFileSettings::getEnergyFileName() { return _energyFile; }

/**
 * @brief get the instant energy file name
 *
 * @return std::string
 */
std::string OutputFileSettings::getInstantEnergyFileName()
{
    return _instEnFile;
}

/**
 * @brief get the momentum file name
 *
 * @return std::string
 */
std::string OutputFileSettings::getMomentumFileName() { return _momFile; }

/**
 * @brief get the trajectory file name
 *
 * @return std::string
 */
std::string OutputFileSettings::getTrajectoryFileName() { return _trajFile; }

/**
 * @brief get the velocity file name
 *
 * @return std::string
 */
std::string OutputFileSettings::getVelocityFileName() { return _velFile; }

/**
 * @brief get the force file name
 *
 * @return std::string
 */
std::string OutputFileSettings::getForceFileName() { return _forceFile; }

/**
 * @brief get the charge file name
 *
 * @return std::string
 */
std::string OutputFileSettings::getChargeFileName() { return _chargeFile; }

/**
 * @brief get the log file name
 *
 * @return std::string
 */
std::string OutputFileSettings::getLogFileName() { return _logFile; }

/**
 * @brief get the info file name
 *
 * @return std::string
 */
std::string OutputFileSettings::getInfoFileName() { return _infoFile; }

/**
 * @brief get the virial file name
 *
 * @return std::string
 */
std::string OutputFileSettings::getVirialFileName() { return _virialFile; }

/**
 * @brief get the stress file name
 *
 * @return std::string
 */
std::string OutputFileSettings::getStressFileName() { return _stressFile; }

/**
 * @brief get the box file name
 *
 * @return std::string
 */
std::string OutputFileSettings::getBoxFileName() { return _boxFile; }

/**
 * @brief get the optimization file name
 *
 * @return std::string
 */
std::string OutputFileSettings::getOptFileName() { return _optFile; }

/**
 * @brief get the ring polymer restart file name
 *
 * @return std::string
 */
std::string OutputFileSettings::getRPMDRestartFileName()
{
    return _rpmdRstFile;
}

/**
 * @brief get the ring polymer trajectory file name
 *
 * @return std::string
 */
std::string OutputFileSettings::getRPMDTrajFileName() { return _rpmdTrajFile; }

/**
 * @brief get the ring polymer velocity file name
 *
 * @return std::string
 */
std::string OutputFileSettings::getRPMDVelocityFileName()
{
    return _rpmdVelFile;
}

/**
 * @brief get the ring polymer force file name
 *
 * @return std::string
 */
std::string OutputFileSettings::getRPMDForceFileName()
{
    return _rpmdForceFile;
}

/**
 * @brief get the ring polymer charge file name
 *
 * @return std::string
 */
std::string OutputFileSettings::getRPMDChargeFileName()
{
    return _rpmdChargeFile;
}

/**
 * @brief get the ring polymer energy file name
 *
 * @return std::string
 */
std::string OutputFileSettings::getRPMDEnergyFileName()
{
    return _rpmdEnergyFile;
}

/**
 * @brief get the timings file name
 *
 * @return std::string
 */
std::string OutputFileSettings::getTimingsFileName() { return _timeFile; }