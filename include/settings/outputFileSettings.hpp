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
    Setting<std::string, "Extended XYZ File">   _extxyzFile;
    Setting<std::string, "H5MD File">           _h5mdFile;
    Setting<std::string, "ASE Trajectory File"> _aseTrajFile;
    Setting<std::string, "Log File">            _logFile{DefaultFiles::logFile};
    Setting<std::string, "Reference File">      _refFile{DefaultFiles::refFile};
    Setting<std::string, "Info File"> _infoFile{DefaultFiles::infoFile};

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
    void setFilePrefixByUser(const std::string &prefix);

    /***************************
     * standard setter methods *
     ***************************/

    void setRestartFileNameByUser(const std::string &);
    void setEnergyFileNameByUser(const std::string &);
    void setInstantEnergyFileNameByUser(const std::string &);
    void setMomentumFileNameByUser(const std::string &);
    void setTrajectoryFileNameByUser(const std::string &);
    void setHybridCenterFileNameByUser(const std::string &);
    void setVelocityFileNameByUser(const std::string &);
    void setForceFileNameByUser(const std::string &);
    void setChargeFileNameByUser(const std::string &);
    void setExtxyzFileNameByUser(const std::string &);
    void setH5mdFileNameByUser(const std::string &);
    void setAseTrajFileNameByUser(const std::string &);
    void setLogFileNameByUser(const std::string &);
    void setRefFileNameByUser(const std::string &);
    void setInfoFileNameByUser(const std::string &);

    void setVirialFileNameByUser(const std::string &);
    void setStressFileNameByUser(const std::string &);
    void setBoxFileNameByUser(const std::string &);

    void setOptFileNameByUser(const std::string &);

    void setRingPolymerRestartFileNameByUser(const std::string &);
    void setRingPolymerTrajectoryFileNameByUser(const std::string &);
    void setRingPolymerVelocityFileNameByUser(const std::string &);
    void setRingPolymerForceFileNameByUser(const std::string &);
    void setRingPolymerChargeFileNameByUser(const std::string &);
    void setRingPolymerEnergyFileNameByUser(const std::string &);

    void setTimingsFileNameByUser(const std::string &);

    void setOverwriteOutputFilesByUser(bool);
    void setIncludeOutputMetadataByUser(bool);
    void setAsyncOutputByUser(bool);
    void setTrajectoryFormatByUser(TrajectoryFormat);

    /***************************
     * standard getter methods *
     ***************************/

    [[nodiscard]] size_t             getOutputFrequency() const;
    [[nodiscard]] const std::string &getFilePrefix() const;

    [[nodiscard]] const std::string &getRestartFileName() const;
    [[nodiscard]] const std::string &getEnergyFileName() const;
    [[nodiscard]] const std::string &getInstantEnergyFileName() const;
    [[nodiscard]] const std::string &getMomentumFileName() const;
    [[nodiscard]] const std::string &getTrajectoryFileName() const;
    [[nodiscard]] const std::string &getHybridCenterFileName() const;
    [[nodiscard]] const std::string &getVelocityFileName() const;
    [[nodiscard]] const std::string &getForceFileName() const;
    [[nodiscard]] const std::string &getChargeFileName() const;
    [[nodiscard]] const std::string &getExtxyzFileName() const;
    [[nodiscard]] const std::string &getH5mdFileName() const;
    [[nodiscard]] const std::string &getAseTrajFileName() const;
    [[nodiscard]] const std::string &getLogFileName() const;
    [[nodiscard]] const std::string &getRefFileName() const;
    [[nodiscard]] const std::string &getInfoFileName() const;

    [[nodiscard]] const std::string &getVirialFileName() const;
    [[nodiscard]] const std::string &getStressFileName() const;
    [[nodiscard]] const std::string &getBoxFileName() const;

    [[nodiscard]] const std::string &getOptFileName() const;

    [[nodiscard]] const std::string &getRPMDRestartFileName() const;
    [[nodiscard]] const std::string &getRPMDTrajFileName() const;
    [[nodiscard]] const std::string &getRPMDVelocityFileName() const;
    [[nodiscard]] const std::string &getRPMDForceFileName() const;
    [[nodiscard]] const std::string &getRPMDChargeFileName() const;
    [[nodiscard]] const std::string &getRPMDEnergyFileName() const;

    [[nodiscard]] const std::string &getTimingsFileName() const;

    [[nodiscard]] bool             getOverwriteOutputFiles() const;
    [[nodiscard]] bool             getIncludeOutputMetadata() const;
    [[nodiscard]] bool             getAsyncOutput() const;
    [[nodiscard]] TrajectoryFormat getTrajectoryFormat() const;

   private:
    [[nodiscard]] std::string _determineMostCommonPrefix() const;

    void _replaceDefaultValues(const std::string &prefix);

    friend struct TestOutputFileSettings;
};

#endif   // SETTINGS__OUTPUT_FILE_SETTINGS_HPP
