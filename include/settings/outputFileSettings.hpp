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
    Setting<size_t> _outputFrequency{1};

    Setting<bool>        _overwriteOutputFiles{false};
    Setting<bool>        _includeOutputMetadata{false};
    Setting<std::string> _filePrefix;

    Setting<std::string> _energyFile{DefaultFiles::energyFile};
    Setting<std::string> _instEnFile{DefaultFiles::instEnFile};
    Setting<std::string> _rstFile{DefaultFiles::restartFile};
    Setting<std::string> _momFile{DefaultFiles::momentumFile};
    Setting<std::string> _trajFile{DefaultFiles::trajFile};
    Setting<std::string> _hybridCenterFile{DefaultFiles::hybridCenterFile};
    Setting<std::string> _velFile{DefaultFiles::velFile};
    Setting<std::string> _forceFile{DefaultFiles::forceFile};
    Setting<std::string> _chargeFile{DefaultFiles::chargeFile};
    Setting<std::string> _logFile{DefaultFiles::logFile};
    Setting<std::string> _refFile{DefaultFiles::refFile};
    Setting<std::string> _infoFile{DefaultFiles::infoFile};

    Setting<std::string> _virialFile{DefaultFiles::virialFile};
    Setting<std::string> _stressFile{DefaultFiles::stressFile};
    Setting<std::string> _boxFile{DefaultFiles::boxFile};

    Setting<std::string> _optFile{DefaultFiles::optFile};

    Setting<std::string> _rpmdRstFile{DefaultFiles::rpmdRstFile};
    Setting<std::string> _rpmdTrajFile{DefaultFiles::rpmdTrajFile};
    Setting<std::string> _rpmdVelFile{DefaultFiles::rpmdVelFile};
    Setting<std::string> _rpmdForceFile{DefaultFiles::rpmdForceFile};
    Setting<std::string> _rpmdChargeFile{DefaultFiles::rpmdChargeFile};
    Setting<std::string> _rpmdEnergyFile{DefaultFiles::rpmdEnergyFile};

    Setting<std::string> _timeFile{DefaultFiles::timingsFile};

   public:
    void finalize();

    void setOutputFrequency(size_t outputFreq);
    void setFilePrefix(const std::string &prefix);

    /***************************
     * standard setter methods *
     ***************************/

    void setRestartFileName(const std::string &);
    void setEnergyFileName(const std::string &);
    void setInstantEnergyFileName(const std::string &);
    void setMomentumFileName(const std::string &);
    void setTrajectoryFileName(const std::string &);
    void setHybridCenterFileName(const std::string &);
    void setVelocityFileName(const std::string &);
    void setForceFileName(const std::string &);
    void setChargeFileName(const std::string &);
    void setLogFileName(const std::string &);
    void setRefFileName(const std::string &);
    void setInfoFileName(const std::string &);

    void setVirialFileName(const std::string &);
    void setStressFileName(const std::string &);
    void setBoxFileName(const std::string &);

    void setOptFileName(const std::string &);

    void setRingPolymerRestartFileName(const std::string &);
    void setRingPolymerTrajectoryFileName(const std::string &);
    void setRingPolymerVelocityFileName(const std::string &);
    void setRingPolymerForceFileName(const std::string &);
    void setRingPolymerChargeFileName(const std::string &);
    void setRingPolymerEnergyFileName(const std::string &);

    void setTimingsFileName(const std::string &);

    void setOverwriteOutputFiles(bool);
    void setIncludeOutputMetadata(bool);

    /***************************
     * standard getter methods *
     ***************************/

    [[nodiscard]] size_t             getOutputFrequency() const;
    [[nodiscard]] const std::string &getFilePrefix() const;
    [[nodiscard]] bool               isFilePrefixSet() const;

    [[nodiscard]] const std::string &getRestartFileName() const;
    [[nodiscard]] const std::string &getEnergyFileName() const;
    [[nodiscard]] const std::string &getInstantEnergyFileName() const;
    [[nodiscard]] const std::string &getMomentumFileName() const;
    [[nodiscard]] const std::string &getTrajectoryFileName() const;
    [[nodiscard]] const std::string &getHybridCenterFileName() const;
    [[nodiscard]] const std::string &getVelocityFileName() const;
    [[nodiscard]] const std::string &getForceFileName() const;
    [[nodiscard]] const std::string &getChargeFileName() const;
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

    [[nodiscard]] bool getOverwriteOutputFiles() const;
    [[nodiscard]] bool getIncludeOutputMetadata() const;

   private:
    [[nodiscard]] std::string _determineMostCommonPrefix() const;

    void _replaceDefaultValues(const std::string &prefix);

    friend struct TestOutputFileSettings;
};

#endif   // SETTINGS__OUTPUT_FILE_SETTINGS_HPP
