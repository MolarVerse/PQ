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

#include "outputInputParser.hpp"

#include <cstddef>

#include "defaults.hpp"
#include "inputKeyAdapter.hpp"
#include "inputRegistry.hpp"
#include "keyMetaData.hpp"
#include "keyRegistry.hpp"
#include "outputFileSettings.hpp"

using namespace input;
using namespace exc;
using namespace settings;

/**
 * @brief Construct a new Input File Parser Output:: Input File Parser Output
 * object
 *
 * @details following keywords are added to the _keywordFuncMap,
 * _keywordRequiredMap and _keywordCountMap:
 * 1)  output_freq "<size_t>"
 * 2)  file_prefix "<string>"
 * 3)  output_file "<string>"
 * 4)  ref_file "<string>"
 * 5)  info_file "<string>"
 * 6)  energy_file "<string>"
 * 7)  instant_energy_file "<string>"
 * 8)  traj_file "<string>"
 * 9)  vel_file "<string>"
 * 10) force_file "<string>"
 * 11) restart_file "<string>"
 * 12) charge_file "<string>"
 * 13) momentum_file "<string>"
 * 14) virial_file "<string>"
 * 15) stress_file "<string>"
 * 16) box_file "<string>"
 * 17) timings_file "<string>"
 * 18) opt_file "<string>"
 * 19) rpmd_restart_file "<string>"
 * 20) rpmd_traj_file "<string>"
 * 21) rpmd_vel_file "<string>"
 * 22) rpmd_force_file "<string>"
 * 23) rpmd_charge_file "<string>"
 * 24) rpmd_energy_file "<string>"
 * 25) include_output_metadata "<bool>"
 */
OutputInputParser::OutputInputParser()
{
    addOutputFrequencyKeyword();
    addFilePrefixKeyword();

    addLogFilenameKeyword();
    addReferenceFilenameKeyword();
    addInfoFilenameKeyword();
    addEnergyFilenameKeyword();
    addInstantEnergyFilenameKeyword();
    addTrajectoryFilenameKeyword();
    addHybridCenterFilenameKeyword();
    addVelocityFilenameKeyword();
    addForceFilenameKeyword();
    addRestartFilenameKeyword();
    addChargeFilenameKeyword();
    addMomentumFilenameKeyword();
    addVirialFilenameKeyword();
    addStressFilenameKeyword();
    addBoxFilenameKeyword();
    addTimingsFilenameKeyword();
    addOptFilenameKeyword();

    addRPMDRestartFilenameKeyword();
    addRPMDTrajectoryFilenameKeyword();
    addRPMDVelocityFilenameKeyword();
    addRPMDForceFilenameKeyword();
    addRPMDChargeFilenameKeyword();
    addRPMDEnergyFilenameKeyword();

    addOverwriteOutputKeyword();
    addIncludeOutputMetadataKeyword();
}

/**
 * @brief add overwrite output keyword to the input parser
 *
 * @details this keyword controls whether existing output files should be
 * overwritten
 */
void OutputInputParser::addOverwriteOutputKeyword()
{
    const auto metaData = KeyMetadata{
        .name        = "overwrite_output",
        .title       = "Overwrite output files",
        .description = "Whether to overwrite existing output files"
    };

    const auto setValue = [](bool value)
    { OutputFileSettings::setOverwriteOutputFiles(value); };

    auto &key = _getRegistry().registerKey(
        KeyRegistry<bool>{
            .metadata     = metaData,
            .defaultValue = false,
            .onSet        = setValue,
        }
    );

    addKeyword(metaData.name, adapt(key), false);
}

/**
 * @brief add include output metadata keyword to the input parser
 *
 * @details this keyword controls whether metadata should be included in output
 * files
 */
void OutputInputParser::addIncludeOutputMetadataKeyword()
{
    const auto metaData = KeyMetadata{
        .name        = "include_output_metadata",
        .title       = "Include output metadata",
        .description = "Whether to include metadata in output files"
    };

    const auto setValue = [](bool value)
    { OutputFileSettings::setIncludeOutputMetadata(value); };

    auto &key = _getRegistry().registerKey(
        KeyRegistry<bool>{
            .metadata     = metaData,
            .defaultValue = false,
            .onSet        = setValue,
        }
    );

    addKeyword(metaData.name, adapt(key), false);
}

/**
 * @brief add output frequency keyword to the input parser
 *
 * @details this keyword controls the frequency at which output files are
 * generated
 */
void OutputInputParser::addOutputFrequencyKeyword()
{
    const auto metaData = KeyMetadata{
        .name        = "output_freq",
        .title       = "Output frequency",
        .description = "Frequency at which output files are generated"
    };

    const auto setValue = [](size_t value)
    { OutputFileSettings::setOutputFrequency(value); };

    auto &key = _getRegistry().registerKey(
        KeyRegistry<size_t>{
            .metadata     = metaData,
            .defaultValue = 1,
            .onSet        = setValue,
        }
    );

    addKeyword(metaData.name, adapt(key), false);
}

/**
 * @brief add file prefix keyword to the input parser
 *
 * @details this keyword controls the prefix for output files
 */
void OutputInputParser::addFilePrefixKeyword()
{
    const auto metaData = KeyMetadata{
        .name        = "file_prefix",
        .title       = "File prefix",
        .description = "Prefix for output files"
    };

    const auto setValue = [](const std::string &value)
    { OutputFileSettings::setFilePrefix(value); };

    auto &key = _getRegistry().registerKey(
        KeyRegistry<std::string>{
            .metadata     = metaData,
            .defaultValue = DefaultFiles::prefix,
            .onSet        = setValue,
        }
    );

    addKeyword(metaData.name, adapt(key), false);
}

/**
 * @brief add reference filename keyword to the input parser
 *
 * @details this keyword controls the filename for the reference output
 */
void OutputInputParser::addLogFilenameKeyword()
{
    const auto metaData = KeyMetadata{
        .name        = "output_file",
        .title       = "Log filename",
        .description = "Filename for the log output"
    };

    const auto setValue = [](const std::string &value)
    { OutputFileSettings::setLogFileName(value); };

    auto &key = _getRegistry().registerKey(
        KeyRegistry<std::string>{
            .metadata     = metaData,
            .defaultValue = DefaultFiles::logFile,
            .onSet        = setValue,
        }
    );

    addKeyword(metaData.name, adapt(key), false);
}

/**
 * @brief add reference filename keyword to the input parser
 *
 * @details this keyword controls the filename for the reference output
 */
void OutputInputParser::addReferenceFilenameKeyword()
{
    const auto metaData = KeyMetadata{
        .name        = "reference_file",
        .title       = "Reference filename",
        .description = "Filename for the reference output"
    };

    const auto setValue = [](const std::string &value)
    { OutputFileSettings::setRefFileName(value); };

    auto &key = _getRegistry().registerKey(
        KeyRegistry<std::string>{
            .metadata     = metaData,
            .defaultValue = DefaultFiles::refFile,
            .onSet        = setValue,
        }
    );

    addKeyword(metaData.name, adapt(key), false);
}

/**
 * @brief add info filename keyword to the input parser
 *
 * @details this keyword controls the filename for the info output
 */
void OutputInputParser::addInfoFilenameKeyword()
{
    const auto metaData = KeyMetadata{
        .name        = "info_file",
        .title       = "Info filename",
        .description = "Filename for the info output"
    };

    const auto setValue = [](const std::string &value)
    { OutputFileSettings::setInfoFileName(value); };

    auto &key = _getRegistry().registerKey(
        KeyRegistry<std::string>{
            .metadata     = metaData,
            .defaultValue = DefaultFiles::infoFile,
            .onSet        = setValue,
        }
    );

    addKeyword(metaData.name, adapt(key), false);
}

/**
 * @brief add energy filename keyword to the input parser
 *
 * @details this keyword controls the filename for the energy output
 */
void OutputInputParser::addEnergyFilenameKeyword()
{
    const auto metaData = KeyMetadata{
        .name        = "energy_file",
        .title       = "Energy filename",
        .description = "Filename for the energy output"
    };

    const auto setValue = [](const std::string &value)
    { OutputFileSettings::setEnergyFileName(value); };

    auto &key = _getRegistry().registerKey(
        KeyRegistry<std::string>{
            .metadata     = metaData,
            .defaultValue = DefaultFiles::energyFile,
            .onSet        = setValue,
        }
    );

    addKeyword(metaData.name, adapt(key), false);
}

/**
 * @brief add instant energy filename keyword to the input parser
 *
 * @details this keyword controls the filename for the instant energy output
 */
void OutputInputParser::addInstantEnergyFilenameKeyword()
{
    const auto metaData = KeyMetadata{
        .name        = "instant_energy_file",
        .title       = "Instant Energy filename",
        .description = "Filename for the instant energy output"
    };

    const auto setValue = [](const std::string &value)
    { OutputFileSettings::setInstantEnergyFileName(value); };

    auto &key = _getRegistry().registerKey(
        KeyRegistry<std::string>{
            .metadata     = metaData,
            .defaultValue = DefaultFiles::instEnFile,
            .onSet        = setValue,
        }
    );

    addKeyword(metaData.name, adapt(key), false);
}

/**
 * @brief add trajectory filename keyword to the input parser
 *
 * @details this keyword controls the filename for the trajectory output
 */
void OutputInputParser::addTrajectoryFilenameKeyword()
{
    const auto metaData = KeyMetadata{
        .name        = "traj_file",
        .title       = "Trajectory filename",
        .description = "Filename for the trajectory output"
    };

    const auto setValue = [](const std::string &value)
    { OutputFileSettings::setTrajectoryFileName(value); };

    auto &key = _getRegistry().registerKey(
        KeyRegistry<std::string>{
            .metadata     = metaData,
            .defaultValue = DefaultFiles::trajFile,
            .onSet        = setValue,
        }
    );

    addKeyword(metaData.name, adapt(key), false);
}

/**
 * @brief add hybrid center filename keyword to the input parser
 *
 * @details this keyword controls the filename for the hybrid center output
 */
void OutputInputParser::addHybridCenterFilenameKeyword()
{
    const auto metaData = KeyMetadata{
        .name        = "hybrid_center_file",
        .title       = "Hybrid Center filename",
        .description = "Filename for the hybrid center output"
    };

    const auto setValue = [](const std::string &value)
    { OutputFileSettings::setHybridCenterFileName(value); };

    auto &key = _getRegistry().registerKey(
        KeyRegistry<std::string>{
            .metadata     = metaData,
            .defaultValue = DefaultFiles::hybridCenterFile,
            .onSet        = setValue,
        }
    );

    addKeyword(metaData.name, adapt(key), false);
}

/**
 * @brief add velocity filename keyword to the input parser
 *
 * @details this keyword controls the filename for the velocity output
 */
void OutputInputParser::addVelocityFilenameKeyword()
{
    const auto metaData = KeyMetadata{
        .name        = "vel_file",
        .title       = "Velocity filename",
        .description = "Filename for the velocity output"
    };

    const auto setValue = [](const std::string &value)
    { OutputFileSettings::setVelocityFileName(value); };

    auto &key = _getRegistry().registerKey(
        KeyRegistry<std::string>{
            .metadata     = metaData,
            .defaultValue = DefaultFiles::velFile,
            .onSet        = setValue,
        }
    );

    addKeyword(metaData.name, adapt(key), false);
}

/**
 * @brief add force filename keyword to the input parser
 *
 * @details this keyword controls the filename for the force output
 */
void OutputInputParser::addForceFilenameKeyword()
{
    const auto metaData = KeyMetadata{
        .name        = "force_file",
        .title       = "Force filename",
        .description = "Filename for the force output"
    };

    const auto setValue = [](const std::string &value)
    { OutputFileSettings::setForceFileName(value); };

    auto &key = _getRegistry().registerKey(
        KeyRegistry<std::string>{
            .metadata     = metaData,
            .defaultValue = DefaultFiles::forceFile,
            .onSet        = setValue,
        }
    );

    addKeyword(metaData.name, adapt(key), false);
}

/**
 * @brief add restart filename keyword to the input parser
 *
 * @details this keyword controls the filename for the restart output
 */
void OutputInputParser::addRestartFilenameKeyword()
{
    const auto metaData = KeyMetadata{
        .name        = "restart_file",
        .title       = "Restart filename",
        .description = "Filename for the restart output"
    };

    const auto setValue = [](const std::string &value)
    { OutputFileSettings::setRestartFileName(value); };

    auto &key = _getRegistry().registerKey(
        KeyRegistry<std::string>{
            .metadata     = metaData,
            .defaultValue = DefaultFiles::restartFile,
            .onSet        = setValue,
        }
    );

    addKeyword(metaData.name, adapt(key), false);
}

/**
 * @brief add charge filename keyword to the input parser
 *
 * @details this keyword controls the filename for the charge output
 */
void OutputInputParser::addChargeFilenameKeyword()
{
    const auto metaData = KeyMetadata{
        .name        = "charge_file",
        .title       = "Charge filename",
        .description = "Filename for the charge output"
    };

    const auto setValue = [](const std::string &value)
    { OutputFileSettings::setChargeFileName(value); };

    auto &key = _getRegistry().registerKey(
        KeyRegistry<std::string>{
            .metadata     = metaData,
            .defaultValue = DefaultFiles::chargeFile,
            .onSet        = setValue,
        }
    );

    addKeyword(metaData.name, adapt(key), false);
}

/**
 * @brief add momentum filename keyword to the input parser
 *
 * @details this keyword controls the filename for the momentum output
 */
void OutputInputParser::addMomentumFilenameKeyword()
{
    const auto metaData = KeyMetadata{
        .name        = "momentum_file",
        .title       = "Momentum filename",
        .description = "Filename for the momentum output"
    };

    const auto setValue = [](const std::string &value)
    { OutputFileSettings::setMomentumFileName(value); };

    auto &key = _getRegistry().registerKey(
        KeyRegistry<std::string>{
            .metadata     = metaData,
            .defaultValue = DefaultFiles::momentumFile,
            .onSet        = setValue,
        }
    );

    addKeyword(metaData.name, adapt(key), false);
}

/**
 * @brief add stress filename keyword to the input parser
 *
 * @details this keyword controls the filename for the stress output
 */
void OutputInputParser::addStressFilenameKeyword()
{
    const auto metaData = KeyMetadata{
        .name        = "stress_file",
        .title       = "Stress filename",
        .description = "Filename for the stress output"
    };

    const auto setValue = [](const std::string &value)
    { OutputFileSettings::setStressFileName(value); };

    auto &key = _getRegistry().registerKey(
        KeyRegistry<std::string>{
            .metadata     = metaData,
            .defaultValue = DefaultFiles::stressFile,
            .onSet        = setValue,
        }
    );

    addKeyword(metaData.name, adapt(key), false);
}

/**
 * @brief add virial filename keyword to the input parser
 *
 * @details this keyword controls the filename for the virial output
 */
void OutputInputParser::addVirialFilenameKeyword()
{
    const auto metaData = KeyMetadata{
        .name        = "virial_file",
        .title       = "Virial filename",
        .description = "Filename for the virial output"
    };

    const auto setValue = [](const std::string &value)
    { OutputFileSettings::setVirialFileName(value); };

    auto &key = _getRegistry().registerKey(
        KeyRegistry<std::string>{
            .metadata     = metaData,
            .defaultValue = DefaultFiles::virialFile,
            .onSet        = setValue,
        }
    );

    addKeyword(metaData.name, adapt(key), false);
}

/**
 * @brief add box filename keyword to the input parser
 *
 * @details this keyword controls the filename for the box output
 */
void OutputInputParser::addBoxFilenameKeyword()
{
    const auto metaData = KeyMetadata{
        .name        = "box_file",
        .title       = "Box filename",
        .description = "Filename for the box output"
    };

    const auto setValue = [](const std::string &value)
    { OutputFileSettings::setBoxFileName(value); };

    auto &key = _getRegistry().registerKey(
        KeyRegistry<std::string>{
            .metadata     = metaData,
            .defaultValue = DefaultFiles::boxFile,
            .onSet        = setValue,
        }
    );

    addKeyword(metaData.name, adapt(key), false);
}

/**
 * @brief add timings filename keyword to the input parser
 *
 * @details this keyword controls the filename for the timings output
 */
void OutputInputParser::addTimingsFilenameKeyword()
{
    const auto metaData = KeyMetadata{
        .name        = "timings_file",
        .title       = "Timings filename",
        .description = "Filename for the timings output"
    };

    const auto setValue = [](const std::string &value)
    { OutputFileSettings::setTimingsFileName(value); };

    auto &key = _getRegistry().registerKey(
        KeyRegistry<std::string>{
            .metadata     = metaData,
            .defaultValue = DefaultFiles::timingsFile,
            .onSet        = setValue,
        }
    );

    addKeyword(metaData.name, adapt(key), false);
}

/**
 * @brief add optimization filename keyword to the input parser
 *
 * @details this keyword controls the filename for the optimization output
 */
void OutputInputParser::addOptFilenameKeyword()
{
    const auto metaData = KeyMetadata{
        .name        = "opt_file",
        .title       = "Optimization filename",
        .description = "Filename for the optimization output"
    };

    const auto setValue = [](const std::string &value)
    { OutputFileSettings::setOptFileName(value); };

    auto &key = _getRegistry().registerKey(
        KeyRegistry<std::string>{
            .metadata     = metaData,
            .defaultValue = DefaultFiles::optFile,
            .onSet        = setValue,
        }
    );

    addKeyword(metaData.name, adapt(key), false);
}

/**
 * @brief parse RPMD restart filename of simulation and add it to output
 *
 * @details default value is default.rpmd.rst
 */
void OutputInputParser::addRPMDRestartFilenameKeyword()
{
    const auto metaData = KeyMetadata{
        .name        = "rpmd_restart_file",
        .title       = "RPMD Restart filename",
        .description = "Filename for the RPMD restart output"
    };

    const auto setValue = [](const std::string &value)
    { OutputFileSettings::setRingPolymerRestartFileName(value); };

    auto &key = _getRegistry().registerKey(
        KeyRegistry<std::string>{
            .metadata     = metaData,
            .defaultValue = DefaultFiles::rpmdRstFile,
            .onSet        = setValue,
        }
    );

    addKeyword(metaData.name, adapt(key), false);
}

/**
 * @brief add RPMD trajectory filename keyword to the input parser
 *
 * @details this keyword controls the filename for the RPMD trajectory output
 */
void OutputInputParser::addRPMDTrajectoryFilenameKeyword()
{
    const auto metaData = KeyMetadata{
        .name        = "rpmd_traj_file",
        .title       = "RPMD Trajectory filename",
        .description = "Filename for the RPMD trajectory output"
    };

    const auto setValue = [](const std::string &value)
    { OutputFileSettings::setRingPolymerTrajectoryFileName(value); };

    auto &key = _getRegistry().registerKey(
        KeyRegistry<std::string>{
            .metadata     = metaData,
            .defaultValue = DefaultFiles::rpmdTrajFile,
            .onSet        = setValue,
        }
    );

    addKeyword(metaData.name, adapt(key), false);
}

/**
 * @brief add RPMD velocity filename keyword to the input parser
 *
 * @details this keyword controls the filename for the RPMD velocity output
 */
void OutputInputParser::addRPMDVelocityFilenameKeyword()
{
    const auto metaData = KeyMetadata{
        .name        = "rpmd_vel_file",
        .title       = "RPMD Velocity filename",
        .description = "Filename for the RPMD velocity output"
    };

    const auto setValue = [](const std::string &value)
    { OutputFileSettings::setRingPolymerVelocityFileName(value); };

    auto &key = _getRegistry().registerKey(
        KeyRegistry<std::string>{
            .metadata     = metaData,
            .defaultValue = DefaultFiles::rpmdVelFile,
            .onSet        = setValue,
        }
    );

    addKeyword(metaData.name, adapt(key), false);
}

/**
 * @brief add RPMD force filename keyword to the input parser
 *
 * @details this keyword controls the filename for the RPMD force output
 */
void OutputInputParser::addRPMDForceFilenameKeyword()
{
    const auto metaData = KeyMetadata{
        .name        = "rpmd_force_file",
        .title       = "RPMD Force filename",
        .description = "Filename for the RPMD force output"
    };

    const auto setValue = [](const std::string &value)
    { OutputFileSettings::setRingPolymerForceFileName(value); };

    auto &key = _getRegistry().registerKey(
        KeyRegistry<std::string>{
            .metadata     = metaData,
            .defaultValue = DefaultFiles::rpmdForceFile,
            .onSet        = setValue,
        }
    );

    addKeyword(metaData.name, adapt(key), false);
}

/**
 * @brief add RPMD charge filename keyword to the input parser
 *
 * @details this keyword controls the filename for the RPMD charge output
 */
void OutputInputParser::addRPMDChargeFilenameKeyword()
{
    const auto metaData = KeyMetadata{
        .name        = "rpmd_charge_file",
        .title       = "RPMD Charge filename",
        .description = "Filename for the RPMD charge output"
    };

    const auto setValue = [](const std::string &value)
    { OutputFileSettings::setRingPolymerChargeFileName(value); };

    auto &key = _getRegistry().registerKey(
        KeyRegistry<std::string>{
            .metadata     = metaData,
            .defaultValue = DefaultFiles::rpmdChargeFile,
            .onSet        = setValue,
        }
    );

    addKeyword(metaData.name, adapt(key), false);
}

/**
 * @brief add RPMD energy filename keyword to the input parser
 *
 * @details this keyword controls the filename for the RPMD energy output
 */
void OutputInputParser::addRPMDEnergyFilenameKeyword()
{
    const auto metaData = KeyMetadata{
        .name        = "rpmd_energy_file",
        .title       = "RPMD Energy filename",
        .description = "Filename for the RPMD energy output"
    };

    const auto setValue = [](const std::string &value)
    { OutputFileSettings::setRingPolymerEnergyFileName(value); };

    auto &key = _getRegistry().registerKey(
        KeyRegistry<std::string>{
            .metadata     = metaData,
            .defaultValue = DefaultFiles::rpmdEnergyFile,
            .onSet        = setValue,
        }
    );

    addKeyword(metaData.name, adapt(key), false);
}
