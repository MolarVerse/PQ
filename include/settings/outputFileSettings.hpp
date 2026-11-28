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

#ifndef SETTINGS__OUTPUT_FILE_SETTINGS_HPP
#define SETTINGS__OUTPUT_FILE_SETTINGS_HPP

#include <cstddef>
#include <string>

#include "defaults.hpp"
#include "enums/output.hpp"
#include "setting.hpp"

struct TestOutputFileSettings;

/**
 * @class OutputFileSettings
 *
 * @brief class to store settings of the output files
 *
 */
class OutputFileSettings
{
   private:
    Setting<size_t, "Output Frequency"> _outputFrequency{1};

    Setting<bool, "Overwrite Output Files">  _overwriteOutputFiles{false};
    Setting<bool, "Include Output Metadata"> _includeOutputMetadata{false};
    Setting<bool, "Async Output">            _asyncOutput{true};
    Setting<TrajectoryFormat, "Trajectory Format"> _trajFormat{
        TrajectoryFormat::XYZ
    };
    Setting<std::string, "File Prefix"> _filePrefix;

    Setting<std::string, "Energy File"> _energyFile{DefaultFiles::energyFile};
    Setting<std::string, "Instant Energy File"> _instEnFile{
        DefaultFiles::instEnFile
    };
    Setting<std::string, "Restart File">  _rstFile{DefaultFiles::restartFile};
    Setting<std::string, "Momentum File"> _momFile{DefaultFiles::momentumFile};
    Setting<std::string, "Trajectory File"> _trajFile{DefaultFiles::trajFile};
    Setting<std::string, "Hybrid Center File"> _hybridCenterFile{
        DefaultFiles::hybridCenterFile
    };
    Setting<std::string, "Velocity File"> _velFile{DefaultFiles::velFile};
    Setting<std::string, "Force File">    _forceFile{DefaultFiles::forceFile};
    Setting<std::string, "Charge File">   _chargeFile{DefaultFiles::chargeFile};
    Setting<std::string, "Extended XYZ File">    _extxyzFile;
    Setting<std::string, "H5MD File">            _h5mdFile;
    Setting<std::string, "ASE Trajectory File">  _aseTrajFile;
    Setting<std::string, "Zarr Trajectory File"> _zarrTrajFile;
    Setting<std::string, "Log File">       _logFile{DefaultFiles::logFile};
    Setting<std::string, "Reference File"> _refFile{DefaultFiles::refFile};
    Setting<std::string, "Info File">      _infoFile{DefaultFiles::infoFile};

    Setting<std::string, "Virial File"> _virialFile{DefaultFiles::virialFile};
    Setting<std::string, "Stress File"> _stressFile{DefaultFiles::stressFile};
    Setting<std::string, "Box File">    _boxFile{DefaultFiles::boxFile};

    Setting<std::string, "Optimization File"> _optFile{DefaultFiles::optFile};

    Setting<std::string, "RPMD Restart File"> _rpmdRstFile{
        DefaultFiles::rpmdRstFile
    };
    Setting<std::string, "RPMD Trajectory File"> _rpmdTrajFile{
        DefaultFiles::rpmdTrajFile
    };
    Setting<std::string, "RPMD Velocity File"> _rpmdVelFile{
        DefaultFiles::rpmdVelFile
    };
    Setting<std::string, "RPMD Force File"> _rpmdForceFile{
        DefaultFiles::rpmdForceFile
    };
    Setting<std::string, "RPMD Charge File"> _rpmdChargeFile{
        DefaultFiles::rpmdChargeFile
    };
    Setting<std::string, "RPMD Energy File"> _rpmdEnergyFile{
        DefaultFiles::rpmdEnergyFile
    };

    Setting<std::string, "Timings File"> _timeFile{DefaultFiles::timingsFile};

   public:
    void finalize();

    void setOutputFrequencyByUser(size_t outputFreq);

    SETTING_GETTER(_outputFrequency);

    SETTING_GETTER_AND_SETTER_BY_USER(_filePrefix);
    SETTING_GETTER_AND_SETTER_BY_USER(_rstFile);
    SETTING_GETTER_AND_SETTER_BY_USER(_energyFile);
    SETTING_GETTER_AND_SETTER_BY_USER(_instEnFile);
    SETTING_GETTER_AND_SETTER_BY_USER(_momFile);
    SETTING_GETTER_AND_SETTER_BY_USER(_trajFile);
    SETTING_GETTER_AND_SETTER_BY_USER(_hybridCenterFile);
    SETTING_GETTER_AND_SETTER_BY_USER(_velFile);
    SETTING_GETTER_AND_SETTER_BY_USER(_forceFile);
    SETTING_GETTER_AND_SETTER_BY_USER(_chargeFile);
    SETTING_GETTER_AND_SETTER_BY_USER(_extxyzFile);
    SETTING_GETTER_AND_SETTER_BY_USER(_h5mdFile);
    SETTING_GETTER_AND_SETTER_BY_USER(_aseTrajFile);
    SETTING_GETTER_AND_SETTER_BY_USER(_zarrTrajFile);
    SETTING_GETTER_AND_SETTER_BY_USER(_logFile);
    SETTING_GETTER_AND_SETTER_BY_USER(_refFile);
    SETTING_GETTER_AND_SETTER_BY_USER(_infoFile);
    SETTING_GETTER_AND_SETTER_BY_USER(_virialFile);
    SETTING_GETTER_AND_SETTER_BY_USER(_stressFile);
    SETTING_GETTER_AND_SETTER_BY_USER(_boxFile);
    SETTING_GETTER_AND_SETTER_BY_USER(_optFile);
    SETTING_GETTER_AND_SETTER_BY_USER(_rpmdRstFile);
    SETTING_GETTER_AND_SETTER_BY_USER(_rpmdTrajFile);
    SETTING_GETTER_AND_SETTER_BY_USER(_rpmdVelFile);
    SETTING_GETTER_AND_SETTER_BY_USER(_rpmdForceFile);
    SETTING_GETTER_AND_SETTER_BY_USER(_rpmdChargeFile);
    SETTING_GETTER_AND_SETTER_BY_USER(_rpmdEnergyFile);
    SETTING_GETTER_AND_SETTER_BY_USER(_timeFile);
    SETTING_GETTER_AND_SETTER_BY_USER(_overwriteOutputFiles);
    SETTING_GETTER_AND_SETTER_BY_USER(_includeOutputMetadata);
    SETTING_GETTER_AND_SETTER_BY_USER(_asyncOutput);
    SETTING_GETTER_AND_SETTER_BY_USER(_trajFormat);

   private:
    [[nodiscard]] std::string _determineMostCommonPrefix() const;

    void _replaceDefaultValues(const std::string &prefix);

    friend struct TestOutputFileSettings;
};

#endif   // SETTINGS__OUTPUT_FILE_SETTINGS_HPP
