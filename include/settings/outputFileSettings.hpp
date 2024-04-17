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

#include "defaults.hpp"

#include <cstddef>       // for size_t
#include <string>        // for string, allocator
#include <string_view>   // for string_view

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

        static inline bool        _filePrefixSet = false;
        static inline std::string _filePrefix;

        static inline std::string _restartFileName       = defaults::_RESTART_FILENAME_DEFAULT_;
        static inline std::string _energyFileName        = defaults::_ENERGY_FILENAME_DEFAULT_;
        static inline std::string _instantEnergyFileName = defaults::_INSTANT_ENERGY_FILENAME_DEFAULT_;
        static inline std::string _momentumFileName      = defaults::_MOMENTUM_FILENAME_DEFAULT_;
        static inline std::string _trajectoryFileName    = defaults::_TRAJECTORY_FILENAME_DEFAULT_;
        static inline std::string _velocityFileName      = defaults::_VELOCITY_FILENAME_DEFAULT_;
        static inline std::string _forceFileName         = defaults::_FORCE_FILENAME_DEFAULT_;
        static inline std::string _chargeFileName        = defaults::_CHARGE_FILENAME_DEFAULT_;
        static inline std::string _logFileName           = defaults::_LOG_FILENAME_DEFAULT_;
        static inline std::string _infoFileName          = defaults::_INFO_FILENAME_DEFAULT_;

        static inline std::string _virialFileName = defaults::_VIRIAL_FILENAME_DEFAULT_;
        static inline std::string _stressFileName = defaults::_STRESS_FILENAME_DEFAULT_;

        static inline std::string _ringPolymerRestartFileName    = defaults::_RING_POLYMER_RESTART_FILENAME_DEFAULT_;
        static inline std::string _ringPolymerTrajectoryFileName = defaults::_RING_POLYMER_TRAJECTORY_FILENAME_DEFAULT_;
        static inline std::string _ringPolymerVelocityFileName   = defaults::_RING_POLYMER_VELOCITY_FILENAME_DEFAULT_;
        static inline std::string _ringPolymerForceFileName      = defaults::_RING_POLYMER_FORCE_FILENAME_DEFAULT_;
        static inline std::string _ringPolymerChargeFileName     = defaults::_RING_POLYMER_CHARGE_FILENAME_DEFAULT_;
        static inline std::string _ringPolymerEnergyFileName     = defaults::_RING_POLYMER_ENERGY_FILENAME_DEFAULT_;

      public:
        OutputFileSettings()  = default;
        ~OutputFileSettings() = default;

        static void setOutputFrequency(const size_t outputFreq);
        static void setFilePrefix(const std::string_view prefix);
        static void replaceDefaultValues(const std::string &prefix);

        [[nodiscard]] static std::string determineMostCommonPrefix();

        [[nodiscard]] static std::string getReferenceFileName();

        /***************************
         * standard setter methods *
         ***************************/

        static void setRestartFileName(const std::string_view name) { _restartFileName = name; }
        static void setEnergyFileName(const std::string_view name) { _energyFileName = name; }
        static void setInstantEnergyFileName(const std::string_view name) { _instantEnergyFileName = name; }
        static void setMomentumFileName(const std::string_view name) { _momentumFileName = name; }
        static void setTrajectoryFileName(const std::string_view name) { _trajectoryFileName = name; }
        static void setVelocityFileName(const std::string_view name) { _velocityFileName = name; }
        static void setForceFileName(const std::string_view name) { _forceFileName = name; }
        static void setChargeFileName(const std::string_view name) { _chargeFileName = name; }
        static void setLogFileName(const std::string_view name) { _logFileName = name; }
        static void setInfoFileName(const std::string_view name) { _infoFileName = name; }
        static void setVirialFileName(const std::string_view name) { _virialFileName = name; }
        static void setStressFileName(const std::string_view name) { _stressFileName = name; }

        static void setRingPolymerRestartFileName(const std::string_view name) { _ringPolymerRestartFileName = name; }
        static void setRingPolymerTrajectoryFileName(const std::string_view name) { _ringPolymerTrajectoryFileName = name; }
        static void setRingPolymerVelocityFileName(const std::string_view name) { _ringPolymerVelocityFileName = name; }
        static void setRingPolymerForceFileName(const std::string_view name) { _ringPolymerForceFileName = name; }
        static void setRingPolymerChargeFileName(const std::string_view name) { _ringPolymerChargeFileName = name; }
        static void setRingPolymerEnergyFileName(const std::string_view name) { _ringPolymerEnergyFileName = name; }

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] static size_t getOutputFrequency() { return _outputFrequency; }

        [[nodiscard]] static bool        isFilePrefixSet() { return _filePrefixSet; }
        [[nodiscard]] static std::string getFilePrefix() { return _filePrefix; }

        [[nodiscard]] static std::string getRestartFileName() { return _restartFileName; }
        [[nodiscard]] static std::string getEnergyFileName() { return _energyFileName; }
        [[nodiscard]] static std::string getInstantEnergyFileName() { return _instantEnergyFileName; }
        [[nodiscard]] static std::string getMomentumFileName() { return _momentumFileName; }
        [[nodiscard]] static std::string getTrajectoryFileName() { return _trajectoryFileName; }
        [[nodiscard]] static std::string getVelocityFileName() { return _velocityFileName; }
        [[nodiscard]] static std::string getForceFileName() { return _forceFileName; }
        [[nodiscard]] static std::string getChargeFileName() { return _chargeFileName; }
        [[nodiscard]] static std::string getLogFileName() { return _logFileName; }
        [[nodiscard]] static std::string getInfoFileName() { return _infoFileName; }
        [[nodiscard]] static std::string getVirialFileName() { return _virialFileName; }
        [[nodiscard]] static std::string getStressFileName() { return _stressFileName; }

        [[nodiscard]] static std::string getRingPolymerRestartFileName() { return _ringPolymerRestartFileName; }
        [[nodiscard]] static std::string getRingPolymerTrajectoryFileName() { return _ringPolymerTrajectoryFileName; }
        [[nodiscard]] static std::string getRingPolymerVelocityFileName() { return _ringPolymerVelocityFileName; }
        [[nodiscard]] static std::string getRingPolymerForceFileName() { return _ringPolymerForceFileName; }
        [[nodiscard]] static std::string getRingPolymerChargeFileName() { return _ringPolymerChargeFileName; }
        [[nodiscard]] static std::string getRingPolymerEnergyFileName() { return _ringPolymerEnergyFileName; }
    };

}   // namespace settings

#endif   // _OUTPUT_FILE_SETTINGS_HPP_