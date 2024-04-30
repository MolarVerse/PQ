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

#ifndef _SETTINGS_HPP_

#define _SETTINGS_HPP_

#include "defaults.hpp"   // for _DIMENSIONALITY_DEFAULT_

#include <string_view>   // for string_view

namespace settings
{
    /**
     * @enum JobType
     *
     * @brief enum class to store the type of the job
     *
     */
    enum class JobType
    {
        MM_MD,
        QM_MD,
        QMMM_MD,
        RING_POLYMER_QM_MD,
        NONE
    };

    /**
     * @class Settings
     *
     * @brief Stores the general settings of the simulation
     *
     */
    class Settings
    {
      private:
        static inline JobType _jobtype;   // no default value

        static inline bool _isMMActivated            = false;
        static inline bool _isQMActivated            = false;
        static inline bool _isRingPolymerMDActivated = false;

        static inline size_t _dimensionality = defaults::_DIMENSIONALITY_DEFAULT_;

      public:
        Settings()  = default;
        ~Settings() = default;

        [[nodiscard]] static bool isQMOnly();

        static void setJobtype(const std::string_view jobtype);
        static void setJobtype(const JobType jobtype) { _jobtype = jobtype; }

        static void activateMM() { _isMMActivated = true; }
        static void activateQM() { _isQMActivated = true; }
        static void activateRingPolymerMD() { _isRingPolymerMDActivated = true; }

        static void deactivateMM() { _isMMActivated = false; }
        static void deactivateQM() { _isQMActivated = false; }
        static void deactivateRingPolymerMD() { _isRingPolymerMDActivated = false; }

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] static JobType getJobtype() { return _jobtype; }

        [[nodiscard]] static size_t getDimensionality() { return _dimensionality; }

        [[nodiscard]] static bool isMMActivated() { return _isMMActivated; }
        [[nodiscard]] static bool isQMActivated() { return _isQMActivated; }
        [[nodiscard]] static bool isQMMMActivated() { return _isMMActivated && _isQMActivated; }
        [[nodiscard]] static bool isQMOnlyActivated() { return _isQMActivated && !_isMMActivated; }
        [[nodiscard]] static bool isMMOnlyActivated() { return _isMMActivated && !_isQMActivated; }
        [[nodiscard]] static bool isRingPolymerMDActivated() { return _isRingPolymerMDActivated; }

        /***************************
         * standard setter methods *
         ***************************/

        static void setIsMMActivated(const bool isMM) { _isMMActivated = isMM; }
        static void setIsQMActivated(const bool isQM) { _isQMActivated = isQM; }
        static void setIsRingPolymerMDActivated(const bool isRingPolymerMD) { _isRingPolymerMDActivated = isRingPolymerMD; }
        static void setDimensionality(const size_t dimensionality) { _dimensionality = dimensionality; }
    };

}   // namespace settings

#endif   // _SETTINGS_HPP_