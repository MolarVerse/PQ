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

#ifndef _OUTPUT_FILE_SETTINGS_HPP_

#define _OUTPUT_FILE_SETTINGS_HPP_

#include <cstddef>       // for size_t
#include <string>        // for string, allocator
#include <string_view>   // for string_view

#include "defaults.hpp"

namespace settings
{
    /**
     * @class OutputFileSettings
     *
     * @brief static class to store settings of the output files
     *
     */
    class OutputFileSettings
    {
       private:
        static inline size_t _outputFrequency = 1;

        static inline bool        _overwriteOutputFiles  = false;
        static inline bool        _includeOutputMetadata = false;
        static inline bool        _filePrefixSet         = false;
        static inline std::string _filePrefix;

        static inline std::string _energyFile = DefaultFiles::energyFile;
        static inline std::string _instEnFile = DefaultFiles::instEnFile;
        static inline std::string _rstFile    = DefaultFiles::restartFile;
        static inline std::string _momFile    = DefaultFiles::momentumFile;
        static inline std::string _trajFile   = DefaultFiles::trajFile;
        static inline std::string _velFile    = DefaultFiles::velFile;
        static inline std::string _forceFile  = DefaultFiles::forceFile;
        static inline std::string _chargeFile = DefaultFiles::chargeFile;
        static inline std::string _logFile    = DefaultFiles::logFile;
        static inline std::string _refFile    = DefaultFiles::refFile;
        static inline std::string _infoFile   = DefaultFiles::infoFile;

        static inline std::string _virialFile = DefaultFiles::virialFile;
        static inline std::string _stressFile = DefaultFiles::stressFile;
        static inline std::string _boxFile    = DefaultFiles::boxFile;

        static inline std::string _optFile = DefaultFiles::optFile;

        // clang-format off
        static inline std::string _rpmdRstFile    = DefaultFiles::rpmdRstFile;
        static inline std::string _rpmdTrajFile   = DefaultFiles::rpmdTrajFile;
        static inline std::string _rpmdVelFile    = DefaultFiles::rpmdVelFile;
        static inline std::string _rpmdForceFile  = DefaultFiles::rpmdForceFile;
        static inline std::string _rpmdChargeFile = DefaultFiles::rpmdChargeFile;
        static inline std::string _rpmdEnergyFile = DefaultFiles::rpmdEnergyFile;
        // clang-format on

        static inline std::string _timeFile = DefaultFiles::timingsFile;

       public:
        OutputFileSettings()  = default;
        ~OutputFileSettings() = default;

        static void setOutputFrequency(const size_t outputFreq);
        static void setFilePrefix(const std::string_view prefix);
        static void replaceDefaultValues(const std::string &prefix);

        [[nodiscard]] static std::string determineMostCommonPrefix();

        /***************************
         * standard setter methods *
         ***************************/

        static void setRestartFileName(const std::string_view);
        static void setEnergyFileName(const std::string_view);
        static void setInstantEnergyFileName(const std::string_view);
        static void setMomentumFileName(const std::string_view);
        static void setTrajectoryFileName(const std::string_view);
        static void setVelocityFileName(const std::string_view);
        static void setForceFileName(const std::string_view);
        static void setChargeFileName(const std::string_view);
        static void setLogFileName(const std::string_view);
        static void setRefFileName(const std::string_view);
        static void setInfoFileName(const std::string_view);

        static void setVirialFileName(const std::string_view);
        static void setStressFileName(const std::string_view);
        static void setBoxFileName(const std::string_view);

        static void setOptFileName(const std::string_view);

        static void setRingPolymerRestartFileName(const std::string_view);
        static void setRingPolymerTrajectoryFileName(const std::string_view);
        static void setRingPolymerVelocityFileName(const std::string_view);
        static void setRingPolymerForceFileName(const std::string_view);
        static void setRingPolymerChargeFileName(const std::string_view);
        static void setRingPolymerEnergyFileName(const std::string_view);

        static void setTimingsFileName(const std::string_view);

        static void setOverwriteOutputFiles(const bool);
        static void setIncludeOutputMetadata(const bool);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] static size_t getOutputFrequency();

        [[nodiscard]] static bool        isFilePrefixSet();
        [[nodiscard]] static std::string getFilePrefix();

        [[nodiscard]] static std::string getRestartFileName();
        [[nodiscard]] static std::string getEnergyFileName();
        [[nodiscard]] static std::string getInstantEnergyFileName();
        [[nodiscard]] static std::string getMomentumFileName();
        [[nodiscard]] static std::string getTrajectoryFileName();
        [[nodiscard]] static std::string getVelocityFileName();
        [[nodiscard]] static std::string getForceFileName();
        [[nodiscard]] static std::string getChargeFileName();
        [[nodiscard]] static std::string getLogFileName();
        [[nodiscard]] static std::string getRefFileName();
        [[nodiscard]] static std::string getInfoFileName();

        [[nodiscard]] static std::string getVirialFileName();
        [[nodiscard]] static std::string getStressFileName();
        [[nodiscard]] static std::string getBoxFileName();

        [[nodiscard]] static std::string getOptFileName();

        [[nodiscard]] static std::string getRPMDRestartFileName();
        [[nodiscard]] static std::string getRPMDTrajFileName();
        [[nodiscard]] static std::string getRPMDVelocityFileName();
        [[nodiscard]] static std::string getRPMDForceFileName();
        [[nodiscard]] static std::string getRPMDChargeFileName();
        [[nodiscard]] static std::string getRPMDEnergyFileName();

        [[nodiscard]] static std::string getTimingsFileName();

        [[nodiscard]] static bool getOverwriteOutputFiles();
        [[nodiscard]] static bool getIncludeOutputMetadata();
    };

}   // namespace settings

#endif   // _OUTPUT_FILE_SETTINGS_HPP_
