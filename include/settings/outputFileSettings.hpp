/*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
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

        static inline std::string _restartFileName    = defaults::_RESTART_FILENAME_DEFAULT_;
        static inline std::string _energyFileName     = defaults::_ENERGY_FILENAME_DEFAULT_;
        static inline std::string _momentumFileName   = defaults::_MOMENTUM_FILENAME_DEFAULT_;
        static inline std::string _trajectoryFileName = defaults::_TRAJECTORY_FILENAME_DEFAULT_;
        static inline std::string _velocityFileName   = defaults::_VELOCITY_FILENAME_DEFAULT_;
        static inline std::string _forceFileName      = defaults::_FORCE_FILENAME_DEFAULT_;
        static inline std::string _chargeFileName     = defaults::_CHARGE_FILENAME_DEFAULT_;
        static inline std::string _logFileName        = defaults::_LOG_FILENAME_DEFAULT_;
        static inline std::string _infoFileName       = defaults::_INFO_FILENAME_DEFAULT_;

        static inline std::string _ringPolymerRestartFileName    = defaults::_RING_POLYMER_RESTART_FILENAME_DEFAULT_;
        static inline std::string _ringPolymerTrajectoryFileName = defaults::_RING_POLYMER_TRAJECTORY_FILENAME_DEFAULT_;
        static inline std::string _ringPolymerVelocityFileName   = defaults::_RING_POLYMER_VELOCITY_FILENAME_DEFAULT_;
        static inline std::string _ringPolymerForceFileName      = defaults::_RING_POLYMER_FORCE_FILENAME_DEFAULT_;
        static inline std::string _ringPolymerChargeFileName     = defaults::_RING_POLYMER_CHARGE_FILENAME_DEFAULT_;

      public:
        OutputFileSettings()  = default;
        ~OutputFileSettings() = default;

        static void setOutputFrequency(const size_t outputFreq);

        /***************************
         * standard setter methods *
         ***************************/

        static void setRestartFileName(const std::string_view name) { OutputFileSettings::_restartFileName = name; }
        static void setEnergyFileName(const std::string_view name) { OutputFileSettings::_energyFileName = name; }
        static void setMomentumFileName(const std::string_view name) { OutputFileSettings::_momentumFileName = name; }
        static void setTrajectoryFileName(const std::string_view name) { OutputFileSettings::_trajectoryFileName = name; }
        static void setVelocityFileName(const std::string_view name) { OutputFileSettings::_velocityFileName = name; }
        static void setForceFileName(const std::string_view name) { OutputFileSettings::_forceFileName = name; }
        static void setChargeFileName(const std::string_view name) { OutputFileSettings::_chargeFileName = name; }
        static void setLogFileName(const std::string_view name) { OutputFileSettings::_logFileName = name; }
        static void setInfoFileName(const std::string_view name) { OutputFileSettings::_infoFileName = name; }

        static void setRingPolymerRestartFileName(const std::string_view name)
        {
            OutputFileSettings::_ringPolymerRestartFileName = name;
        }
        static void setRingPolymerTrajectoryFileName(const std::string_view name)
        {
            OutputFileSettings::_ringPolymerTrajectoryFileName = name;
        }
        static void setRingPolymerVelocityFileName(const std::string_view name)
        {
            OutputFileSettings::_ringPolymerVelocityFileName = name;
        }
        static void setRingPolymerForceFileName(const std::string_view name)
        {
            OutputFileSettings::_ringPolymerForceFileName = name;
        }
        static void setRingPolymerChargeFileName(const std::string_view name)
        {
            OutputFileSettings::_ringPolymerChargeFileName = name;
        }

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] static size_t getOutputFrequency() { return OutputFileSettings::_outputFrequency; }

        [[nodiscard]] static std::string getRestartFileName() { return OutputFileSettings::_restartFileName; }
        [[nodiscard]] static std::string getEnergyFileName() { return OutputFileSettings::_energyFileName; }
        [[nodiscard]] static std::string getMomentumFileName() { return OutputFileSettings::_momentumFileName; }
        [[nodiscard]] static std::string getTrajectoryFileName() { return OutputFileSettings::_trajectoryFileName; }
        [[nodiscard]] static std::string getVelocityFileName() { return OutputFileSettings::_velocityFileName; }
        [[nodiscard]] static std::string getForceFileName() { return OutputFileSettings::_forceFileName; }
        [[nodiscard]] static std::string getChargeFileName() { return OutputFileSettings::_chargeFileName; }
        [[nodiscard]] static std::string getLogFileName() { return OutputFileSettings::_logFileName; }
        [[nodiscard]] static std::string getInfoFileName() { return OutputFileSettings::_infoFileName; }

        [[nodiscard]] static std::string getRingPolymerRestartFileName()
        {
            return OutputFileSettings::_ringPolymerRestartFileName;
        }
        [[nodiscard]] static std::string getRingPolymerTrajectoryFileName()
        {
            return OutputFileSettings::_ringPolymerTrajectoryFileName;
        }
        [[nodiscard]] static std::string getRingPolymerVelocityFileName()
        {
            return OutputFileSettings::_ringPolymerVelocityFileName;
        }
        [[nodiscard]] static std::string getRingPolymerForceFileName() { return OutputFileSettings::_ringPolymerForceFileName; }
        [[nodiscard]] static std::string getRingPolymerChargeFileName() { return OutputFileSettings::_ringPolymerChargeFileName; }
    };

}   // namespace settings

#endif   // _OUTPUT_FILE_SETTINGS_HPP_